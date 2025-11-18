package ru.sr.app.ui

import android.content.Context
import androidx.lifecycle.ViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import ru.sr.core.crypto.ICryptoService
import ru.sr.core.mesh.PeerPresence
import ru.sr.core.mesh.PresenceStore
import ru.sr.core.mesh.Proximity
import ru.sr.app.R
import ru.sr.core.storage.MessageRepository
import ru.sr.core.storage.SettingsRepository
import ru.sr.core.storage.db.MessageEntity
import ru.sr.core.testing.DispatcherProvider
import ru.sr.core.transport.ServiceIdGenerator
import ru.sr.core.transport.TransportCoordinator
import kotlin.random.Random

@HiltViewModel
class SRAppViewModel @Inject constructor(
    private val settingsRepository: SettingsRepository,
    private val messageRepository: MessageRepository,
    private val cryptoService: ICryptoService,
    private val presenceStore: PresenceStore,
    private val transportCoordinator: TransportCoordinator,
    private val dispatcherProvider: DispatcherProvider,
    @ApplicationContext private val context: Context
) : ViewModel() {

    private val _state = MutableStateFlow(SRAppState())
    val state: StateFlow<SRAppState> = _state.asStateFlow()

    private val _commands = MutableSharedFlow<SRCommand>(extraBufferCapacity = 8)
    val commands: SharedFlow<SRCommand> = _commands.asSharedFlow()
    private var activeConversationJob: Job? = null

    init {
        bootstrapMockData()
        observeSettings()
        observePeers()
        observeChats()
    }

    private fun bootstrapMockData() {
        viewModelScope.launch(dispatcherProvider.default) {
            listOf(
                PeerPresence("peer-1", "Ольга", 1, Proximity.NEAR, System.currentTimeMillis(), true, true),
                PeerPresence("peer-2", "Радиоузел", 2, Proximity.WALL, System.currentTimeMillis(), false, true),
                PeerPresence("peer-3", "Команда", 3, Proximity.FAR, System.currentTimeMillis(), true, false)
            ).forEach { presenceStore.update(it) }
        }
        viewModelScope.launch(dispatcherProvider.io) {
            val sampleMessage = MessageEntity(
                id = "msg-${System.currentTimeMillis()}",
                conversationId = "peer-1",
                senderId = "peer-1",
                cipherText = context.getString(R.string.sample_message_hello).encodeToByteArray(),
                timestamp = System.currentTimeMillis(),
                delivered = true,
                ttl = 6
            )
            messageRepository.save(sampleMessage)
        }
    }

    private fun observeSettings() {
        viewModelScope.launch {
            combine(
                settingsRepository.nickname(),
                settingsRepository.visible(),
                settingsRepository.zone(),
                settingsRepository.topic(),
                settingsRepository.powerProfile()
            ) { nickname, visible, zone, topic, power ->
                _state.update { state ->
                    state.copy(
                        onboarding = state.onboarding.copy(isVisible = visible),
                        home = state.home.copy(
                            zone = zone,
                            topic = topic,
                            profile = state.home.profile.copy(
                                nickname = nickname,
                                fingerprint = cryptoService.fingerprintFor(nickname),
                                visible = visible,
                                zone = zone,
                                topic = topic,
                                powerProfile = when (power) {
                                    0 -> PowerProfile.TEXT
                                    1 -> PowerProfile.BALANCED
                                    else -> PowerProfile.FAST
                                }
                            )
                        )
                    )
                }
            }.collect { }
        }
    }

    private fun observePeers() {
        viewModelScope.launch {
            presenceStore.peers.collect { list ->
                val peers = list.map { presence ->
                    PeerUi(
                        peerId = presence.peerId,
                        nickname = presence.nickname.orEmpty(),
                        hops = presence.hops,
                        proximity = presence.proximity,
                        isOnline = presence.online,
                        supportsDirect = presence.supportsDirect
                    )
                }
                _state.update { it.copy(home = it.home.copy(peers = peers)) }
            }
        }
    }

    private fun observeChats() {
        viewModelScope.launch {
            messageRepository.observeConversations().collect { conversations ->
                val chats = conversations.map { projection ->
                    ChatSummaryUi(
                        conversationId = projection.conversationId,
                        title = projection.conversationId,
                        lastMessage = "",
                        delivered = true
                    )
                }
                _state.update { it.copy(home = it.home.copy(chats = chats)) }
            }
        }
    }

    fun onEvent(event: SRAppEvent) {
        when (event) {
            SRAppEvent.OnOnboardingNext -> advanceOnboarding()
            SRAppEvent.OnOnboardingSkip -> _state.update { it.copy(showOnboarding = false) }
            SRAppEvent.OnStart -> {
                _state.update { it.copy(showOnboarding = false) }
                viewModelScope.launch { _commands.emit(SRCommand.StartMesh) }
                viewModelScope.launch(dispatcherProvider.io) {
                    val advertisingToken = buildServiceId(_state.value.home.zone, _state.value.home.topic)
                    val requireFallback = !_state.value.onboarding.bluetoothGranted || !_state.value.onboarding.wifiGranted || !_state.value.onboarding.serviceGranted
                    if (requireFallback) {
                        transportCoordinator.start(advertisingToken, useFallback = true)
                    } else {
                        runCatching {
                            transportCoordinator.start(advertisingToken, useFallback = false)
                        }.onFailure {
                            transportCoordinator.start(advertisingToken, useFallback = true)
                        }
                    }
                }
            }
            is SRAppEvent.OnToggleVisible -> viewModelScope.launch { settingsRepository.setVisible(event.value) }
            is SRAppEvent.OnZoneChanged -> viewModelScope.launch { settingsRepository.setZone(event.zone) }
            is SRAppEvent.OnTopicChanged -> viewModelScope.launch { settingsRepository.setTopic(event.topic) }
            is SRAppEvent.OnSelectTab -> _state.update { it.copy(home = it.home.copy(selectedTab = event.tab)) }
            is SRAppEvent.OnSelectChat -> selectChat(event.conversationId)
            is SRAppEvent.OnSendMessage -> sendMessage(event.peerId, event.message)
            is SRAppEvent.OnChangePower -> viewModelScope.launch { settingsRepository.setPowerProfile(event.power.ordinal) }
            is SRAppEvent.OnPermissionResult -> updatePermissionState(event)
        }
    }


    private fun buildServiceId(zone: String, topic: String): String {
        return ServiceIdGenerator.generate(zone, topic)
    }

    private fun selectChat(conversationId: String) {
        _state.update { current ->
            current.copy(home = current.home.copy(selectedChatId = conversationId))
        }
        activeConversationJob?.cancel()
        activeConversationJob = viewModelScope.launch {
            messageRepository.observe(conversationId).collect { entities ->
                val items = entities.map { entity ->
                    MessageUi(
                        id = entity.id,
                        authorId = entity.senderId,
                        text = entity.cipherText.decodeToString(),
                        delivered = entity.delivered
                    )
                }
                _state.update { updated ->
                    updated.copy(home = updated.home.copy(selectedChatId = conversationId, messages = items))
                }
            }
        }
    }

    private fun updatePermissionState(event: SRAppEvent.OnPermissionResult) {
        _state.update { current ->
            current.copy(
                onboarding = current.onboarding.copy(
                    bluetoothGranted = event.bluetooth ?: current.onboarding.bluetoothGranted,
                    wifiGranted = event.wifi ?: current.onboarding.wifiGranted,
                    serviceGranted = event.service ?: current.onboarding.serviceGranted
                )
            )
        }
    }

    private fun advanceOnboarding() {
        val newIndex = (_state.value.onboarding.currentPage + 1).coerceAtMost(_state.value.onboarding.pages.lastIndex)
        _state.update { it.copy(onboarding = it.onboarding.copy(currentPage = newIndex)) }
    }

    private fun sendMessage(peerId: String, message: String) {
        viewModelScope.launch(dispatcherProvider.io) {
            val entity = MessageEntity(
                id = "local-${System.currentTimeMillis()}-${Random.nextInt(1000)}",
                conversationId = peerId,
                senderId = "local",
                cipherText = message.encodeToByteArray(),
                timestamp = System.currentTimeMillis(),
                delivered = false,
                ttl = 6
            )
            messageRepository.save(entity)
            presenceStore.update(
                PeerPresence(peerId, null, 1, Proximity.NEAR, System.currentTimeMillis(), true, true)
            )
        }
    }
}

package ru.sr.app.ui

import androidx.annotation.StringRes
import ru.sr.app.R
import ru.sr.core.mesh.Proximity
data class SRAppState(
    val showOnboarding: Boolean = true,
    val onboarding: OnboardingState = OnboardingState(),
    val home: HomeState = HomeState()
)

data class OnboardingState(
    val pages: List<OnboardingPage> = listOf(
        OnboardingPage(0, titleRes = R.string.onboarding_title_discovery, descriptionRes = R.string.onboarding_body_discovery),
        OnboardingPage(1, titleRes = R.string.onboarding_title_permissions, descriptionRes = R.string.onboarding_body_permissions),
        OnboardingPage(2, titleRes = R.string.onboarding_title_privacy, descriptionRes = R.string.onboarding_body_privacy)
    ),
    val currentPage: Int = 0,
    val isVisible: Boolean = true,
    val bluetoothGranted: Boolean = false,
    val wifiGranted: Boolean = false,
    val serviceGranted: Boolean = false
)

data class OnboardingPage(
    val index: Int,
    @StringRes val titleRes: Int,
    @StringRes val descriptionRes: Int
)

data class HomeState(
    val selectedTab: HomeTab = HomeTab.PEOPLE,
    val zone: String = "ОФИС-5Э",
    val topic: String = "DEFAULT",
    val onlyContacts: Boolean = false,
    val peers: List<PeerUi> = emptyList(),
    val chats: List<ChatSummaryUi> = emptyList(),
    val selectedChatId: String? = null,
    val messages: List<MessageUi> = emptyList(),
    val profile: ProfileUi = ProfileUi()
)

data class PeerUi(
    val peerId: String,
    val nickname: String,
    val hops: Int,
    val proximity: Proximity,
    val isOnline: Boolean,
    val supportsDirect: Boolean
)

data class ChatSummaryUi(
    val conversationId: String,
    val title: String,
    val lastMessage: String,
    val delivered: Boolean
)

data class MessageUi(
    val id: String,
    val authorId: String,
    val text: String,
    val delivered: Boolean
)

data class ProfileUi(
    val nickname: String = "Пользователь",
    val fingerprint: String = "",
    val visible: Boolean = true,
    val zone: String = "ОФИС-5Э",
    val topic: String = "DEFAULT",
    val powerProfile: PowerProfile = PowerProfile.BALANCED
)

enum class HomeTab { PEOPLE, CHATS, PROFILE }

enum class PowerProfile { TEXT, BALANCED, FAST }

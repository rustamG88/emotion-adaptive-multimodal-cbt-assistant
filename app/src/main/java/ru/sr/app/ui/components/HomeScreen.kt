package ru.sr.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import ru.sr.app.R
import ru.sr.app.ui.HomeState
import ru.sr.app.ui.HomeTab
import ru.sr.app.ui.MessageUi
import ru.sr.app.ui.PeerUi
import ru.sr.app.ui.PowerProfile
import ru.sr.app.ui.ProfileUi
import ru.sr.app.ui.SRAppEvent
import ru.sr.core.mesh.Proximity
import ru.sr.core.ui.components.SRPrimaryButton

@Composable
fun HomeScreen(state: HomeState, onEvent: (SRAppEvent) -> Unit) {
    Column(modifier = Modifier.fillMaxSize()) {
        when (state.selectedTab) {
            HomeTab.PEOPLE -> PeopleScreen(state, onEvent)
            HomeTab.CHATS -> ChatList(state, onEvent)
            HomeTab.PROFILE -> ProfileScreen(state.profile, onEvent)
        }
        NavigationBar {
            NavigationBarItem(
                selected = state.selectedTab == HomeTab.PEOPLE,
                onClick = { onEvent(SRAppEvent.OnSelectTab(HomeTab.PEOPLE)) },
                label = { Text(text = stringResource(id = R.string.tab_people)) },
                icon = {}
            )
            NavigationBarItem(
                selected = state.selectedTab == HomeTab.CHATS,
                onClick = { onEvent(SRAppEvent.OnSelectTab(HomeTab.CHATS)) },
                label = { Text(text = stringResource(id = R.string.tab_chats)) },
                icon = {}
            )
            NavigationBarItem(
                selected = state.selectedTab == HomeTab.PROFILE,
                onClick = { onEvent(SRAppEvent.OnSelectTab(HomeTab.PROFILE)) },
                label = { Text(text = stringResource(id = R.string.tab_profile)) },
                icon = {}
            )
        }
    }
}

@Composable
private fun PeopleScreen(state: HomeState, onEvent: (SRAppEvent) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 24.dp)
    ) {
        Text(text = stringResource(id = R.string.tab_people), style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        FilterRow(state, onEvent)
        Spacer(modifier = Modifier.height(16.dp))
        if (state.peers.isEmpty()) {
            Text(text = stringResource(id = R.string.people_empty_state), style = MaterialTheme.typography.bodyLarge)
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                items(state.peers) { peer ->
                    PeerCard(peer) {
                        onEvent(SRAppEvent.OnSelectChat(peer.peerId))
                        onEvent(SRAppEvent.OnSelectTab(HomeTab.CHATS))
                    }
                }
            }
        }
    }
}

@Composable
private fun FilterRow(state: HomeState, onEvent: (SRAppEvent) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        TextField(
            value = state.zone,
            onValueChange = { onEvent(SRAppEvent.OnZoneChanged(it)) },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(text = stringResource(id = R.string.nearby_zone_label)) },
            colors = TextFieldDefaults.colors()
        )
        TextField(
            value = state.topic,
            onValueChange = { onEvent(SRAppEvent.OnTopicChanged(it)) },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(text = stringResource(id = R.string.nearby_topic_label)) },
            colors = TextFieldDefaults.colors()
        )
    }
}

@Composable
private fun PeerCard(peer: PeerUi, onOpenChat: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    val displayName = if (peer.nickname.isBlank()) stringResource(id = R.string.default_nickname) else peer.nickname
                    Text(text = displayName, style = MaterialTheme.typography.titleLarge)
                    val status = if (peer.isOnline) stringResource(id = R.string.nearby_status_online) else stringResource(id = R.string.nearby_status_offline)
                    val proximityLabel = when (peer.proximity) {
                        Proximity.NEAR -> stringResource(id = R.string.nearby_proximity_near)
                        Proximity.WALL -> stringResource(id = R.string.nearby_proximity_wall)
                        Proximity.FAR -> stringResource(id = R.string.nearby_proximity_far)
                    }
                    Text(
                        text = status + " • " + proximityLabel + " • " + stringResource(id = R.string.people_status_reachable, peer.hops),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                if (peer.supportsDirect) {
                    Text(
                        text = stringResource(id = R.string.nearby_supports_direct),
                        style = MaterialTheme.typography.labelLarge,
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.secondary, MaterialTheme.shapes.small)
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
            SRPrimaryButton(
                text = stringResource(id = R.string.tab_chats),
                modifier = Modifier.fillMaxWidth(),
                onClick = onOpenChat
            )
        }
    }
}

@Composable
private fun ChatList(state: HomeState, onEvent: (SRAppEvent) -> Unit) {
    Column(modifier = Modifier.padding(16.dp)) {
        Text(text = stringResource(id = R.string.chat_list_title), style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(state.chats) { chat ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(text = chat.title, style = MaterialTheme.typography.titleLarge)
                        val preview = if (chat.lastMessage.isBlank()) stringResource(id = R.string.chat_last_message_placeholder) else chat.lastMessage
                        Text(text = preview, style = MaterialTheme.typography.bodyLarge)
                        Spacer(modifier = Modifier.height(12.dp))
                        SRPrimaryButton(
                            text = stringResource(id = R.string.tab_chats),
                            modifier = Modifier.fillMaxWidth(),
                            onClick = { onEvent(SRAppEvent.OnSelectChat(chat.conversationId)) }
                        )
                    }
                }
            }
        }
        state.selectedChatId?.let { conversationId ->
            Spacer(modifier = Modifier.height(24.dp))
            ChatDetail(
                conversationId = conversationId,
                messages = state.messages,
                onSend = { text -> onEvent(SRAppEvent.OnSendMessage(conversationId, text)) }
            )
        }
    }
}

@Composable
private fun ChatDetail(conversationId: String, messages: List<MessageUi>, onSend: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
        Text(text = stringResource(id = R.string.chat_dialog_title, conversationId), style = MaterialTheme.typography.titleLarge)
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.height(200.dp)) {
            items(messages) { message ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.surfaceVariant, MaterialTheme.shapes.medium)
                        .padding(12.dp)
                ) {
                    val authorLabel = if (message.authorId == "local") stringResource(id = R.string.chat_author_me) else message.authorId
                    Text(text = authorLabel, style = MaterialTheme.typography.labelLarge)
                    Text(text = message.text, style = MaterialTheme.typography.bodyLarge)
                    val status = if (message.delivered) stringResource(id = R.string.chat_delivered) else stringResource(id = R.string.chat_sending)
                    Text(text = status, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        val draft = remember { mutableStateOf("") }
        TextField(
            value = draft.value,
            onValueChange = { draft.value = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(text = stringResource(id = R.string.chat_message_hint)) },
            colors = TextFieldDefaults.colors()
        )
        SRPrimaryButton(
            text = stringResource(id = R.string.chat_send),
            modifier = Modifier.fillMaxWidth(),
            onClick = {
                if (draft.value.isNotBlank()) {
                    onSend(draft.value)
                    draft.value = ""
                }
            }
        )
        Text(text = stringResource(id = R.string.chat_encryption_banner), style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun ProfileScreen(profile: ProfileUi, onEvent: (SRAppEvent) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(text = stringResource(id = R.string.profile_title), style = MaterialTheme.typography.headlineMedium)
        Text(text = stringResource(id = R.string.profile_my_nickname) + ": " + profile.nickname)
        Text(text = stringResource(id = R.string.profile_my_id) + ": " + profile.fingerprint)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Text(text = stringResource(id = R.string.profile_toggle_visible))
            Switch(checked = profile.visible, onCheckedChange = { onEvent(SRAppEvent.OnToggleVisible(it)) })
        }
        TextField(
            value = profile.zone,
            onValueChange = { onEvent(SRAppEvent.OnZoneChanged(it)) },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(text = stringResource(id = R.string.profile_default_zone)) },
            colors = TextFieldDefaults.colors()
        )
        TextField(
            value = profile.topic,
            onValueChange = { onEvent(SRAppEvent.OnTopicChanged(it)) },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(text = stringResource(id = R.string.profile_default_topic)) },
            colors = TextFieldDefaults.colors()
        )
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(text = stringResource(id = R.string.profile_power_modes), fontWeight = FontWeight.Bold)
            PowerProfile.values().forEach { mode ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    val label = when (mode) {
                        PowerProfile.TEXT -> stringResource(id = R.string.profile_power_text)
                        PowerProfile.BALANCED -> stringResource(id = R.string.profile_power_balanced)
                        PowerProfile.FAST -> stringResource(id = R.string.profile_power_fast)
                    }
                    Text(text = label)
                    RadioButton(
                        selected = profile.powerProfile == mode,
                        onClick = { onEvent(SRAppEvent.OnChangePower(mode)) }
                    )
                }
            }
            Text(text = stringResource(id = R.string.profile_energy_notes), style = MaterialTheme.typography.bodySmall)
        }
    }
}

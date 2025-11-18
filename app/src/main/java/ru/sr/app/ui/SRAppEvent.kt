package ru.sr.app.ui

sealed class SRAppEvent {
    object OnOnboardingNext : SRAppEvent()
    object OnOnboardingSkip : SRAppEvent()
    object OnStart : SRAppEvent()
    data class OnToggleVisible(val value: Boolean) : SRAppEvent()
    data class OnZoneChanged(val zone: String) : SRAppEvent()
    data class OnTopicChanged(val topic: String) : SRAppEvent()
    data class OnSelectTab(val tab: HomeTab) : SRAppEvent()
    data class OnSelectChat(val conversationId: String) : SRAppEvent()
    data class OnSendMessage(val peerId: String, val message: String) : SRAppEvent()
    data class OnChangePower(val power: PowerProfile) : SRAppEvent()
    data class OnPermissionResult(val bluetooth: Boolean? = null, val wifi: Boolean? = null, val service: Boolean? = null) : SRAppEvent()
}

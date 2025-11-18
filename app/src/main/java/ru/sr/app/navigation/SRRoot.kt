package ru.sr.app.navigation

import androidx.compose.runtime.Composable
import ru.sr.app.ui.SRAppEvent
import ru.sr.app.ui.SRAppState
import ru.sr.app.ui.components.HomeScreen
import ru.sr.app.ui.components.OnboardingScreen

@Composable
fun SRRoot(
    state: SRAppState,
    onEvent: (SRAppEvent) -> Unit
) {
    if (state.showOnboarding) {
        OnboardingScreen(state.onboarding, onEvent)
    } else {
        HomeScreen(state.home, onEvent)
    }
}

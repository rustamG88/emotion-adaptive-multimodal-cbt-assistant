package ru.sr.app.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import ru.sr.app.R
import ru.sr.app.ui.OnboardingState
import ru.sr.app.ui.SRAppEvent
import ru.sr.core.ui.components.SRPrimaryButton

@Composable
fun OnboardingScreen(
    state: OnboardingState,
    onEvent: (SRAppEvent) -> Unit
) {
    val page = state.pages[state.currentPage]
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp, vertical = 32.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = stringResource(id = R.string.splash_logo_text),
                style = MaterialTheme.typography.displayLarge,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = stringResource(id = R.string.splash_subtitle),
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.secondary
            )
            Spacer(modifier = Modifier.height(40.dp))
            Text(
                text = stringResource(id = page.titleRes),
                style = MaterialTheme.typography.headlineMedium,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = stringResource(id = page.descriptionRes),
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(24.dp))
            PermissionCard(state, onEvent)
        }
        Column(modifier = Modifier.fillMaxWidth()) {
            SRPrimaryButton(
                text = if (state.currentPage == state.pages.lastIndex) stringResource(id = R.string.onboarding_start) else stringResource(id = R.string.onboarding_next),
                modifier = Modifier.fillMaxWidth(),
                onClick = {
                    if (state.currentPage == state.pages.lastIndex) {
                        onEvent(SRAppEvent.OnStart)
                    } else {
                        onEvent(SRAppEvent.OnOnboardingNext)
                    }
                }
            )
            Spacer(modifier = Modifier.height(12.dp))
            SRPrimaryButton(
                text = stringResource(id = R.string.onboarding_skip),
                modifier = Modifier.fillMaxWidth(),
                onClick = { onEvent(SRAppEvent.OnOnboardingSkip) }
            )
        }
    }
}

@Composable
private fun PermissionCard(state: OnboardingState, onEvent: (SRAppEvent) -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(PaddingValues(horizontal = 16.dp, vertical = 20.dp)),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(text = stringResource(id = R.string.onboarding_permission_rationale), style = MaterialTheme.typography.bodyLarge)
            ToggleRow(
                title = stringResource(id = R.string.onboarding_toggle_visible),
                checked = state.isVisible,
                onCheckedChange = { onEvent(SRAppEvent.OnToggleVisible(it)) }
            )
            ToggleRow(
                title = stringResource(id = R.string.onboarding_request_bluetooth),
                checked = state.bluetoothGranted,
                onCheckedChange = { onEvent(SRAppEvent.OnPermissionResult(bluetooth = it)) }
            )
            ToggleRow(
                title = stringResource(id = R.string.onboarding_request_wifi),
                checked = state.wifiGranted,
                onCheckedChange = { onEvent(SRAppEvent.OnPermissionResult(wifi = it)) }
            )
            ToggleRow(
                title = stringResource(id = R.string.onboarding_request_service),
                checked = state.serviceGranted,
                onCheckedChange = { onEvent(SRAppEvent.OnPermissionResult(service = it)) }
            )
        }
    }
}

@Composable
private fun ToggleRow(title: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    androidx.compose.foundation.layout.Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = title, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

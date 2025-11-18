package ru.sr.app

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dagger.hilt.android.AndroidEntryPoint
import ru.sr.app.navigation.SRRoot
import ru.sr.app.service.MeshForegroundService
import ru.sr.app.ui.SRAppViewModel
import ru.sr.app.ui.SRCommand
import ru.sr.core.ui.theme.SRTheme

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private val viewModel: SRAppViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SRTheme {
                val context = LocalContext.current
                val state by viewModel.state.collectAsStateWithLifecycle()
                LaunchedEffect(Unit) {
                    viewModel.commands.collect { command ->
                        when (command) {
                            SRCommand.StartMesh -> {
                                val intent = Intent(context, MeshForegroundService::class.java)
                                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                                    context.startForegroundService(intent)
                                } else {
                                    context.startService(intent)
                                }
                            }
                        }
                    }
                }
                SRRoot(
                    state = state,
                    onEvent = viewModel::onEvent
                )
            }
        }
    }
}

package ru.sr.core.transport.mock

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import ru.sr.core.transport.MeshTransport
import ru.sr.core.transport.TransportPayload
import ru.sr.core.transport.TransportState

class MockTransport : MeshTransport {
    private val scope = CoroutineScope(Dispatchers.Default)
    private val stateFlow = MutableStateFlow(TransportState(false, 0, false))
    private val payloadFlow = MutableSharedFlow<TransportPayload>()

    override val state = stateFlow.asStateFlow()
    override val payloads = payloadFlow.asSharedFlow()

    override suspend fun start(advertisingToken: String) {
        scope.launch {
            stateFlow.emit(TransportState(true, 2, false))
            repeat(3) { index ->
                delay(1500L * (index + 1))
                payloadFlow.emit(
                    TransportPayload(
                        peerId = "mock-$index",
                        data = "heartbeat".encodeToByteArray(),
                        reliable = false
                    )
                )
            }
        }
    }

    override suspend fun stop() {
        stateFlow.emit(TransportState(false, 0, false))
    }

    override suspend fun send(payload: TransportPayload) {
        scope.launch {
            payloadFlow.emit(payload)
        }
    }
}

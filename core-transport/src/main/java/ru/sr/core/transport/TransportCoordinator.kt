package ru.sr.core.transport

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import ru.sr.core.transport.mock.MockTransport

class TransportCoordinator(
    private val primary: MeshTransport,
    private val fallback: MeshTransport = MockTransport()
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _state = MutableStateFlow(TransportState(false, 0, false))
    private val _payloads = MutableSharedFlow<TransportPayload>()
    private var stateJob: Job? = null
    private var payloadJob: Job? = null
    private var activeTransport: MeshTransport? = null

    val state: Flow<TransportState> = _state.asStateFlow()
    val payloads: Flow<TransportPayload> = _payloads.asSharedFlow()

    suspend fun start(advertisingToken: String, useFallback: Boolean) {
        val transport = if (useFallback) fallback else primary
        if (activeTransport != transport) {
            cancelCollectors()
            activeTransport = transport
        } else {
            cancelCollectors()
        }

        stateJob = scope.launch {
            transport.state.collect { state ->
                _state.emit(state)
            }
        }
        payloadJob = scope.launch {
            transport.payloads.collect { payload ->
                _payloads.emit(payload)
            }
        }

        try {
            transport.start(advertisingToken)
        } catch (throwable: Throwable) {
            cancelCollectors()
            activeTransport = null
            _state.emit(TransportState(false, 0, false))
            throw throwable
        }
    }

    suspend fun stop(useFallback: Boolean) {
        val transport = activeTransport ?: if (useFallback) fallback else primary
        try {
            transport.stop()
        } finally {
            cancelCollectors()
            activeTransport = null
            _state.emit(TransportState(false, 0, false))
        }
    }

    suspend fun send(payload: TransportPayload, useFallback: Boolean) {
        val transport = activeTransport ?: if (useFallback) fallback else primary
        transport.send(payload)
    }

    private fun cancelCollectors() {
        stateJob?.cancel()
        payloadJob?.cancel()
        stateJob = null
        payloadJob = null
    }
}

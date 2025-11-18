package ru.sr.core.transport.nearby

import android.content.Context
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.connection.AdvertisingOptions
import com.google.android.gms.nearby.connection.ConnectionLifecycleCallback
import com.google.android.gms.nearby.connection.ConnectionsClient
import com.google.android.gms.nearby.connection.PayloadCallback
import com.google.android.gms.nearby.connection.Strategy
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import ru.sr.core.transport.MeshTransport
import ru.sr.core.transport.TransportPayload
import ru.sr.core.transport.TransportState

class NearbyTransport(context: Context) : MeshTransport {
    private val client: ConnectionsClient = Nearby.getConnectionsClient(context)
    private val scope = CoroutineScope(Dispatchers.IO)
    private val stateFlow = MutableStateFlow(TransportState(false, 0, true))
    private val payloadFlow = MutableSharedFlow<TransportPayload>()

    override val state = stateFlow.asStateFlow()
    override val payloads = payloadFlow.asSharedFlow()

    private val lifecycleCallback = object : ConnectionLifecycleCallback() {}
    private val payloadCallback = object : PayloadCallback() {}

    override suspend fun start(advertisingToken: String) {
        scope.launch {
            stateFlow.emit(stateFlow.value.copy(isRunning = true))
        }
        val options = AdvertisingOptions.Builder().setStrategy(Strategy.P2P_STAR).build()
        client.startAdvertising(ENDPOINT_NAME, advertisingToken, lifecycleCallback, options)
    }

    override suspend fun stop() {
        client.stopAllEndpoints()
        client.stopAdvertising()
        scope.launch {
            stateFlow.emit(TransportState(false, 0, true))
        }
    }

    override suspend fun send(payload: TransportPayload) {
        // Production implementation will route payload to endpoint
        // For scaffold we emit locally to simulate delivery
        scope.launch {
            payloadFlow.emit(payload)
        }
    }

    companion object {
        private const val ENDPOINT_NAME = "SR"
    }
}

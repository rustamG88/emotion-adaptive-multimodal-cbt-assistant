package ru.sr.core.transport

import kotlinx.coroutines.flow.Flow

interface MeshTransport {
    val state: Flow<TransportState>
    val payloads: Flow<TransportPayload>
    suspend fun start(advertisingToken: String)
    suspend fun stop()
    suspend fun send(payload: TransportPayload)
}

data class TransportState(
    val isRunning: Boolean,
    val nearbyCount: Int,
    val supportsDirect: Boolean
)

data class TransportPayload(
    val peerId: String,
    val data: ByteArray,
    val reliable: Boolean
)

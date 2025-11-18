package ru.sr.core.mesh

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlin.random.Random

class HeartbeatManager(private val scope: CoroutineScope) {
    private val heartbeats = MutableSharedFlow<Heartbeat>(extraBufferCapacity = 16)
    private var job: Job? = null

    fun observe(): Flow<Heartbeat> = heartbeats.asSharedFlow()

    fun start(ephemeralId: String) {
        if (job?.isActive == true) return
        job = scope.launch {
            while (true) {
                val jitter = Random.nextLong(0, 3000)
                delay(12_000L + jitter)
                heartbeats.emit(Heartbeat(ephemeralId, System.currentTimeMillis()))
            }
        }
    }

    fun stop() {
        job?.cancel()
    }
}

data class Heartbeat(
    val ephemeralId: String,
    val timestamp: Long
)

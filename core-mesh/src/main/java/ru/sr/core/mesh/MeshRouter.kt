package ru.sr.core.mesh

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.min

class MeshRouter(
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default),
    private val cacheSize: Int = 20_000,
    private val ackTimeoutMillis: Long = 4_000L
) {
    private val seenFrames = object : LinkedHashMap<String, Long>(cacheSize, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Long>?): Boolean {
            return size > cacheSize
        }
    }
    private val ackObservers = MutableSharedFlow<MeshFrame>(extraBufferCapacity = 32)
    private val pending = ConcurrentHashMap<String, Pending>()

    val acknowledgements: Flow<MeshFrame> = ackObservers.asSharedFlow()

    fun shouldForward(frame: MeshFrame): Boolean {
        synchronized(seenFrames) {
            val exists = seenFrames.containsKey(frame.header.id)
            if (!exists && frame.header.ttl > 0) {
                seenFrames[frame.header.id] = System.currentTimeMillis()
                return true
            }
            return false
        }
    }

    fun record(frame: MeshFrame) {
        synchronized(seenFrames) {
            seenFrames[frame.header.id] = System.currentTimeMillis()
        }
    }

    fun trackDelivery(frame: MeshFrame, reliable: Boolean) {
        if (!reliable) return
        val job = scope.launch {
            var attempts = 0
            var delayMs = ackTimeoutMillis
            while (attempts < MAX_ATTEMPTS) {
                delay(delayMs)
                val pendingMessage = pending[frame.header.id] ?: break
                pendingMessage.onRetry(frame.nextRetry(attempts + 1))
                attempts += 1
                delayMs = min(delayMs * 2, MAX_BACKOFF)
            }
            pending.remove(frame.header.id)
        }
        pending[frame.header.id] = Pending(job)
    }

    fun ackReceived(frameId: String) {
        pending.remove(frameId)?.cancel()
    }

    fun buildAck(forFrame: MeshFrame): MeshFrame = MeshFrame(
        header = MeshHeader(
            id = UUID.randomUUID().toString(),
            ttl = 3,
            hop = 0,
            frameClass = FrameClass.ACK,
            topicTag = forFrame.header.topicTag
        ),
        body = forFrame.header.id.encodeToByteArray(),
        signature = byteArrayOf()
    )

    private fun MeshFrame.nextRetry(attempt: Int): MeshFrame = copy(
        header = header.copy(hop = 0, ttl = 6),
        signature = signature
    ).also {
        scope.launch { ackObservers.emit(it) }
    }

    private class Pending(private val job: Job) {
        fun cancel() {
            job.cancel()
        }

        fun onRetry(frame: MeshFrame) {
            // Emission handled by router's flow in nextRetry
        }
    }

    companion object {
        private const val MAX_ATTEMPTS = 3
        private const val MAX_BACKOFF = 16_000L
    }
}

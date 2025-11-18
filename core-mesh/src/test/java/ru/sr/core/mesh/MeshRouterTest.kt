package ru.sr.core.mesh

import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class MeshRouterTest {
    @Test
    fun `router deduplicates frames`() {
        val router = MeshRouter()
        val frame = MeshFrame(MeshHeader(), byteArrayOf(), byteArrayOf())
        assertTrue(router.shouldForward(frame))
        assertFalse(router.shouldForward(frame))
    }

    @Test
    fun `router respects ttl`() {
        val router = MeshRouter()
        val frame = MeshFrame(MeshHeader(ttl = 0), byteArrayOf(), byteArrayOf())
        assertFalse(router.shouldForward(frame))
    }

    @Test
    fun `router retries reliable frames`() = runBlocking {
        val router = MeshRouter(this, cacheSize = 10, ackTimeoutMillis = 10)
        val frame = MeshFrame(MeshHeader(), byteArrayOf(), byteArrayOf())
        val job = launch { router.acknowledgements.first() }
        router.trackDelivery(frame, reliable = true)
        job.join()
    }
}

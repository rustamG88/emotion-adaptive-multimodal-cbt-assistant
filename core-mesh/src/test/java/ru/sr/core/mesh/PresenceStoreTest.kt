package ru.sr.core.mesh

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PresenceStoreTest {
    @Test
    fun `store updates and sorts peers`() = runBlocking {
        val store = PresenceStore()
        store.update(PeerPresence("b", "Борис", 1, Proximity.NEAR, 0L, true, true))
        store.update(PeerPresence("a", "Алексей", 1, Proximity.NEAR, 0L, true, true))
        val peers = store.peers.first()
        assertEquals(listOf("Алексей", "Борис"), peers.map { it.nickname })
    }

    @Test
    fun `store prunes stale`() = runBlocking {
        val store = PresenceStore()
        store.update(PeerPresence("old", null, 1, Proximity.FAR, 0L, false, false))
        store.prune(1000L, now = 2000L)
        val peers = store.peers.first()
        assertTrue(peers.isEmpty())
    }

    @Test
    fun `proximity classification`() {
        val store = PresenceStore()
        assertEquals(Proximity.NEAR, store.proximityFromRssi(-50))
        assertEquals(Proximity.WALL, store.proximityFromRssi(-70))
        assertEquals(Proximity.FAR, store.proximityFromRssi(-90))
    }
}

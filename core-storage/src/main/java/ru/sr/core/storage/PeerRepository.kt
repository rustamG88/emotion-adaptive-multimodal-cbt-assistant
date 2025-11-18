package ru.sr.core.storage

import kotlinx.coroutines.flow.Flow
import ru.sr.core.storage.db.PeerDao
import ru.sr.core.storage.db.PeerEntity

class PeerRepository(private val peerDao: PeerDao) {
    fun observePeers(): Flow<List<PeerEntity>> = peerDao.observePeers()

    suspend fun upsert(peer: PeerEntity) {
        peerDao.upsert(peer)
    }

    suspend fun cleanup(threshold: Long) {
        peerDao.removeStale(threshold)
    }
}

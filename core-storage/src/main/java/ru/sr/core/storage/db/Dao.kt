package ru.sr.core.storage.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity)

    @Query("SELECT * FROM messages WHERE conversationId = :conversationId ORDER BY timestamp ASC")
    fun observeConversation(conversationId: String): Flow<List<MessageEntity>>


    @Query("SELECT conversationId, MAX(timestamp) AS lastTimestamp FROM messages GROUP BY conversationId ORDER BY lastTimestamp DESC")
    fun observeConversations(): Flow<List<ConversationProjection>>
}

@Dao
interface PeerDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(peer: PeerEntity)

    @Query("SELECT * FROM peers")
    fun observePeers(): Flow<List<PeerEntity>>

    @Query("DELETE FROM peers WHERE lastSeen < :threshold")
    suspend fun removeStale(threshold: Long)
}

@Dao
interface SessionDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(session: SessionEntity)

    @Query("SELECT * FROM sessions WHERE peerId = :peerId")
    suspend fun get(peerId: String): SessionEntity?
}

@Dao
interface DeviceDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(device: DeviceEntity)

    @Query("SELECT * FROM devices")
    fun observeDevices(): Flow<List<DeviceEntity>>

    @Query("DELETE FROM devices WHERE id = :id")
    suspend fun delete(id: String)
}


data class ConversationProjection(
    val conversationId: String,
    val lastTimestamp: Long
)

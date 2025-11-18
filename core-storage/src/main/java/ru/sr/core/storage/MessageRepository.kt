package ru.sr.core.storage

import kotlinx.coroutines.flow.Flow
import ru.sr.core.storage.db.ConversationProjection
import ru.sr.core.storage.db.MessageDao
import ru.sr.core.storage.db.MessageEntity

class MessageRepository(private val messageDao: MessageDao) {
    fun observe(conversationId: String): Flow<List<MessageEntity>> = messageDao.observeConversation(conversationId)

    fun observeConversations(): Flow<List<ConversationProjection>> = messageDao.observeConversations()

    suspend fun save(message: MessageEntity) {
        messageDao.insert(message)
    }
}

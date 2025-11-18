package ru.sr.core.storage.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(
    entities = [MessageEntity::class, PeerEntity::class, SessionEntity::class, DeviceEntity::class],
    version = 1
)
@TypeConverters(ByteArrayConverter::class)
abstract class SRDatabase : RoomDatabase() {
    abstract fun messageDao(): MessageDao
    abstract fun peerDao(): PeerDao
    abstract fun sessionDao(): SessionDao
    abstract fun deviceDao(): DeviceDao
}

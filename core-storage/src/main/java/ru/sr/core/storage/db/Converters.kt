package ru.sr.core.storage.db

import androidx.room.TypeConverter

class ByteArrayConverter {
    @TypeConverter
    fun fromByteArray(value: ByteArray?): String? = value?.joinToString(separator = ",") { it.toString() }

    @TypeConverter
    fun toByteArray(value: String?): ByteArray? = value?.split(',')?.mapNotNull { it.toIntOrNull()?.toByte() }?.toByteArray()
}

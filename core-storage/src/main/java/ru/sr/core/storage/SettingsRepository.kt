package ru.sr.core.storage

import kotlinx.coroutines.flow.Flow
import ru.sr.core.storage.datastore.SettingsDataStore

class SettingsRepository(private val store: SettingsDataStore) {
    fun nickname(): Flow<String> = store.nickname()
    suspend fun setNickname(value: String) = store.setNickname(value)

    fun zone(): Flow<String> = store.zone()
    suspend fun setZone(value: String) = store.setZone(value)

    fun topic(): Flow<String> = store.topic()
    suspend fun setTopic(value: String) = store.setTopic(value)

    fun visible(): Flow<Boolean> = store.visible()
    suspend fun setVisible(value: Boolean) = store.setVisible(value)

    fun powerProfile(): Flow<Int> = store.powerProfile()
    suspend fun setPowerProfile(value: Int) = store.setPowerProfile(value)
}

package ru.sr.core.storage.datastore

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "sr_settings")

class SettingsDataStore(private val context: Context) {
    private object Keys {
        val nickname = stringPreferencesKey("nickname")
        val zone = stringPreferencesKey("zone")
        val topic = stringPreferencesKey("topic")
        val visible = booleanPreferencesKey("visible")
        val power = intPreferencesKey("power")
    }

    fun nickname(): Flow<String> = context.dataStore.data.map { prefs ->
        prefs[Keys.nickname] ?: ""
    }

    suspend fun setNickname(value: String) {
        context.dataStore.edit { prefs ->
            prefs[Keys.nickname] = value
        }
    }

    fun zone(): Flow<String> = context.dataStore.data.map { prefs ->
        prefs[Keys.zone] ?: "ОФИС-5Э"
    }

    suspend fun setZone(value: String) {
        context.dataStore.edit { prefs ->
            prefs[Keys.zone] = value
        }
    }

    fun topic(): Flow<String> = context.dataStore.data.map { prefs ->
        prefs[Keys.topic] ?: "DEFAULT"
    }

    suspend fun setTopic(value: String) {
        context.dataStore.edit { prefs ->
            prefs[Keys.topic] = value
        }
    }

    fun visible(): Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[Keys.visible] ?: true
    }

    suspend fun setVisible(value: Boolean) {
        context.dataStore.edit { prefs ->
            prefs[Keys.visible] = value
        }
    }

    fun powerProfile(): Flow<Int> = context.dataStore.data.map { prefs ->
        prefs[Keys.power] ?: 1
    }

    suspend fun setPowerProfile(value: Int) {
        context.dataStore.edit { prefs ->
            prefs[Keys.power] = value
        }
    }
}

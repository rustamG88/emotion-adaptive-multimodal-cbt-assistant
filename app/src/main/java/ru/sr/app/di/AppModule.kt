package ru.sr.app.di

import android.content.Context
import androidx.room.Room
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import ru.sr.core.crypto.FakeCryptoService
import ru.sr.core.crypto.ICryptoService
import ru.sr.core.crypto.IKeyStore
import ru.sr.core.mesh.MeshRouter
import ru.sr.core.mesh.PresenceStore
import ru.sr.core.storage.MessageRepository
import ru.sr.core.storage.PeerRepository
import ru.sr.core.storage.SettingsRepository
import ru.sr.core.storage.datastore.SettingsDataStore
import ru.sr.core.storage.db.SRDatabase
import ru.sr.core.testing.DefaultDispatcherProvider
import ru.sr.core.testing.DispatcherProvider
import ru.sr.core.transport.TransportCoordinator
import ru.sr.core.transport.nearby.NearbyTransport

@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Provides
    @Singleton
    fun provideDispatcherProvider(): DispatcherProvider = DefaultDispatcherProvider()

    @Provides
    @Singleton
    fun provideApplicationScope(): CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): SRDatabase = Room.databaseBuilder(
        context,
        SRDatabase::class.java,
        "sr.db"
    ).fallbackToDestructiveMigration().build()

    @Provides
    @Singleton
    fun provideSettingsDataStore(@ApplicationContext context: Context): SettingsDataStore = SettingsDataStore(context)

    @Provides
    @Singleton
    fun provideSettingsRepository(store: SettingsDataStore): SettingsRepository = SettingsRepository(store)

    @Provides
    @Singleton
    fun providePeerRepository(database: SRDatabase): PeerRepository = PeerRepository(database.peerDao())

    @Provides
    @Singleton
    fun provideMessageRepository(database: SRDatabase): MessageRepository = MessageRepository(database.messageDao())

    @Provides
    @Singleton
    fun provideFakeCrypto(): FakeCryptoService = FakeCryptoService()

    @Provides
    @Singleton
    fun provideCrypto(service: FakeCryptoService): ICryptoService = service

    @Provides
    @Singleton
    fun provideKeyStore(service: FakeCryptoService): IKeyStore = service

    @Provides
    @Singleton
    fun providePresenceStore(): PresenceStore = PresenceStore()

    @Provides
    @Singleton
    fun provideRouter(scope: CoroutineScope): MeshRouter = MeshRouter(scope)

    @Provides
    @Singleton
    fun provideTransportCoordinator(@ApplicationContext context: Context): TransportCoordinator {
        return TransportCoordinator(NearbyTransport(context))
    }
}

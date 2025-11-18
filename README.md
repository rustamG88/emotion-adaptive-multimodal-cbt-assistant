# SR — Связь рядом

SR is an offline-first mesh messenger built with Kotlin and Jetpack Compose. The application discovers peers nearby, establishes secure end-to-end encrypted channels, and keeps working even when the Internet is unavailable. This repository contains the full Android client implementation with a modular architecture and test coverage for the core routing primitives.

## Project structure

```
SR/
├── app/                  # Android application module with Compose UI, navigation, DI wiring, foreground service
├── core-ui/              # Design system (Material 3 theme, typography, shared components)
├── core-transport/       # Nearby Connections transport, Wi-Fi Direct hooks, mock transport for fallback
├── core-mesh/            # Mesh router, frame model, heartbeat and presence store
├── core-crypto/          # Cryptography interfaces, Fake/Signal adapter scaffolding
├── core-storage/         # Room entities/DAOs, repositories, DataStore settings, SQLCipher hooks (commented)
├── core-permissions/     # Runtime permission coordinator for Android 12–15 requirements
├── core-testing/         # Shared test utilities (dispatcher providers, common assertions)
└── gradle/               # Version catalog and Gradle wrapper configuration
```

## Build configuration

* **JDK:** 17
* **Kotlin:** 2.0.20
* **Android Gradle Plugin:** 8.6.0
* **compileSdk / targetSdk:** 35
* **minSdk:** 26
* **Compose BOM:** 2024.10.00
* **Detekt / ktlint:** enabled for all modules with strict settings

Debug builds use the `dev` flavor with the `FakeCryptoService` and mock transport fallback. Release builds enable R8 + resource shrinking and are wired for a future Signal protocol adapter and SQLCipher storage (dependencies commented for opt-in).

## Modules

| Module | Responsibility |
| --- | --- |
| `app` | Compose navigation, splash, onboarding, foreground service, state orchestration, Hilt DI |
| `core-ui` | Material 3 theme tuned for Cyrillic, SR brand palette, reusable Compose components |
| `core-transport` | Nearby Connections client wrapper, Wi-Fi Direct scaffold, mock transport for graceful degradation |
| `core-mesh` | Mesh router with TTL enforcement, deduplication, retry/ACK handling, presence & heartbeat management |
| `core-crypto` | Cryptography contracts (X3DH/Double Ratchet abstraction), fake crypto engine, libsignal adapter stub |
| `core-storage` | Room entities & DAOs (`Message`, `Peer`, `Session`, `Device`), repositories, DataStore-backed settings |
| `core-permissions` | Typed permission coordinator covering Bluetooth, Nearby Wi-Fi, foreground service, notifications |
| `core-testing` | Dispatcher provider implementations and common testing helpers |

## Running the project

1. Ensure Android Studio (Ladybug or newer) with Android SDK 26–35 is installed.
2. Clone the repository and open it in Android Studio.
3. Select the `app` run configuration and the `devDebug` build variant.
4. Deploy to an emulator or device. On first launch the SR splash animates, onboarding (in Russian) appears, and mock peers are displayed in the “Люди рядом” screen if real transports are unavailable.

> **Note:** The provided `gradlew` script delegates to a locally installed Gradle distribution. Install Gradle 8.9 (or newer compatible) to build from the command line.

## Permissions and Android 12–15 support

The onboarding flow educates users about the required permissions and exposes toggles for:

* `BLUETOOTH_SCAN`, `BLUETOOTH_ADVERTISE`, `BLUETOOTH_CONNECT`
* `NEARBY_WIFI_DEVICES`
* `FOREGROUND_SERVICE_DATA_SYNC` / `FOREGROUND_SERVICE_CONNECTED_DEVICE`
* `POST_NOTIFICATIONS`

`core-permissions` contains helpers that respect `neverForLocation` flags where applicable, and the `MeshForegroundService` uses the `dataSync` foreground service type with a persistent notification (“Связь рядом активна”).

## Cryptography & transport strategy

* Identity keys rely on Android Keystore (scaffolded via `IKeyStore`).
* `FakeCryptoService` ships for the `dev` flavor; swap to the real Signal adapter by implementing `LibSignalAdapter` and wiring dependencies in `core-crypto`.
* Frames follow a minimal header (UUID, TTL, hop, class) plus AEAD ciphertext and Ed25519 signature placeholders.
* `MeshRouter` enforces TTL, deduplicates via LRU cache, and retries reliable frames with exponential backoff.
* `TransportCoordinator` orchestrates Nearby Connections and the mock transport fallback, allowing graceful operation when radios are unavailable.

## Feature flags & configuration

* **Wi-Fi Direct upgrade:** scaffolded in `core-transport`, disabled by default.
* **Media transfer:** reserved frame classes (10/11) and TODO hooks for future implementation.
* **Power profiles:** selectable in the profile screen (`Текст`, `Сбалансированный`, `Макс. скорость`) and persisted in DataStore.

## Testing

Unit tests live in `core-mesh` (router TTL/dedup/backoff, presence store) and `core-testing`. Execute with:

```bash
./gradlew test
```

Instrumented tests can be expanded in `app/src/androidTest`. A baseline service smoke test scaffold is included via the foreground service startup.

## Switching crypto implementations

* **Fake (default):** Provided by `FakeCryptoService`, perfect for development and UI work.
* **Signal (planned):** Uncomment the `libsignal-client` dependency in `core-crypto`, implement `LibSignalAdapter`, and bind it via Hilt. The interfaces (`ICryptoService`, `ISession`, `IGroupSession`) are ready for the Signal/X3DH + Double Ratchet workflow.

## Security notes

* All messages travel through encrypted frames (AEAD placeholder in dev builds).
* Heartbeats use rotating ephemeral IDs and avoid leaking nicknames before mutual handshake.
* Replay protection and deduplication are enforced by `MeshRouter`’s LRU cache.
* Storage can be hardened with SQLCipher by enabling the commented dependency and migration hooks.

## License

The project is provided as-is for demonstration and evaluation of offline mesh messaging patterns.

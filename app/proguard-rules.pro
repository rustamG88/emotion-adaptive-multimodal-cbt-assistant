# Keep Hilt generated components
-keep class dagger.hilt.internal.** { *; }
-keep class * extends dagger.hilt.android.internal.lifecycle.HiltViewModelFactory$ViewModelFactoriesEntryPoint { *; }

# Room schema
-keep class androidx.room.** { *; }
-keep interface androidx.room.** { *; }

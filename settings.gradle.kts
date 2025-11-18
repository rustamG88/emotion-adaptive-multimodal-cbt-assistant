pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven("https://plugins.gradle.org/m2/")
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
    versionCatalogs {
        create("libs") {
            val tomlFile = file("gradle/libs.versions.toml")
            val versions = mutableMapOf<String, String>()
            data class LibrarySpec(val alias: String, val module: String, val version: String?, val versionRef: String?)
            data class PluginSpec(val alias: String, val id: String, val version: String?, val versionRef: String?)
            val libraries = linkedMapOf<String, LibrarySpec>()
            val plugins = linkedMapOf<String, PluginSpec>()
            var section = ""

            tomlFile.forEachLine { rawLine ->
                val line = rawLine.trim()
                if (line.isEmpty() || line.startsWith("#")) return@forEachLine
                if (line.startsWith("[")) {
                    section = line.removePrefix("[").removeSuffix("]")
                    return@forEachLine
                }
                when (section) {
                    "versions" -> {
                        val parts = line.split("=", limit = 2)
                        if (parts.size == 2) {
                            versions[parts[0].trim()] = parts[1].trim().trim('"')
                        }
                    }
                    "libraries" -> parseCatalogEntry(line)?.let { (alias, attributes) ->
                        val module = attributes["module"] ?: return@let
                        val version = attributes["version"]
                        val versionRef = attributes["version.ref"]
                        libraries[alias] = LibrarySpec(alias, module, version, versionRef)
                    }
                    "plugins" -> parseCatalogEntry(line)?.let { (alias, attributes) ->
                        val id = attributes["id"] ?: return@let
                        val version = attributes["version"]
                        val versionRef = attributes["version.ref"]
                        plugins[alias] = PluginSpec(alias, id, version, versionRef)
                    }
                }
            }

            versions.forEach { (alias, value) ->
                version(alias, value)
            }
            libraries.values.forEach { spec ->
                val (group, name) = spec.module.split(":", limit = 2)
                val dependency = library(spec.alias, group, name)
                spec.versionRef?.let { dependency.versionRef(it) } ?: spec.version?.let { dependency.version(it) }
            }
            plugins.values.forEach { spec ->
                val plugin = plugin(spec.alias, spec.id)
                spec.versionRef?.let { plugin.versionRef(it) } ?: spec.version?.let { plugin.version(it) }
            }
        }
    }
}

fun parseCatalogEntry(line: String): Pair<String, Map<String, String>>? {
    val parts = line.split("=", limit = 2)
    if (parts.size != 2) return null
    val alias = parts[0].trim()
    val content = parts[1].substringAfter("{").substringBeforeLast("}")
    if (content.isBlank()) return null
    val attributes = content.split(",")
        .mapNotNull { attribute ->
            val keyValue = attribute.split("=", limit = 2)
            if (keyValue.size != 2) return@mapNotNull null
            keyValue[0].trim() to keyValue[1].trim().trim('"')
        }
        .toMap()
    return alias to attributes
}

rootProject.name = "SR"

include(
    ":app",
    ":core-ui",
    ":core-transport",
    ":core-mesh",
    ":core-crypto",
    ":core-storage",
    ":core-permissions",
    ":core-testing"
)

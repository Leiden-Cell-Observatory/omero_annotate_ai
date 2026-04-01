pluginManagement {
    repositories {
        gradlePluginPortal()
        maven("https://maven.scijava.org/content/repositories/releases")
        maven("https://maven.scijava.org/content/repositories/snapshots")
    }
}

plugins {
    id("io.github.qupath.platform") version "0.2.0"
}

rootProject.name = "qupath-extension-omero-annotate"

gradle.extra["qupath.app.version"] = "0.6.0"

gradle.extra["qupath.extension.name"] = "OMERO Annotate AI"
gradle.extra["qupath.extension.group"] = "io.github.leiden-cell-observatory"
gradle.extra["qupath.extension.version"] = "0.1.0-SNAPSHOT"
gradle.extra["qupath.extension.description"] = "Create AI training data from QuPath annotations with optional OMERO integration"
gradle.extra["qupath.extension.automaticModule"] = "io.github.leidencellobs.qupath.ext.omero.annotate"

plugins {
    id("qupath-conventions")
    `java-library`
}

qupathExtension {
    name = "OMERO Annotate AI"
    version = "0.1.0-SNAPSHOT"
    group = "io.github.leiden-cell-observatory"
    description = "Create AI training data from QuPath annotations with optional OMERO integration"
    automaticModule = "io.github.leidencellobs.qupath.ext.omero.annotate"
}

repositories {
    mavenCentral()
    maven("https://maven.scijava.org/content/repositories/releases")
    maven("https://maven.scijava.org/content/repositories/snapshots")
    maven("https://artifacts.openmicroscopy.org/artifactory/maven/")
}

dependencies {
    // QuPath (provided by the platform)
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)

    // YAML configuration (Jackson)
    implementation("com.fasterxml.jackson.dataformat:jackson-dataformat-yaml:2.17.2")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.2")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.17.2")

    // OMERO (compileOnly - optional at runtime for local-only mode)
    compileOnly("org.openmicroscopy:omero-gateway:5.8.3")
    compileOnly("org.openmicroscopy:omero-blitz:5.7.1")

    // Testing
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

tasks.test {
    useJUnitPlatform()
}

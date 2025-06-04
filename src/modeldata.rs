use crate::cards::street::Street;
use crate::mccfr::nlhe::encoder::BlueprintEncoder;
use crate::mccfr::nlhe::Profile;
use crate::save::disk::Disk;

/// Generate model training data from a profile
pub struct ModelDataGenerator {
    profile: Profile,
    encoder: BlueprintEncoder,
}

impl ModelDataGenerator {
    /// Create a new model data generator
    pub fn new() -> Self {
        log::info!("Loading profile for model data generation");
        let profile = Profile::load_parquet();
        log::info!("Loading encoder for abstraction lookups");
        let encoder = BlueprintEncoder::load(Street::Pref); // Load preflop encoder
        Self { profile, encoder }
    }

    /// Generate model training data and save to parquet file
    pub fn generate(&self) {
        // Delegate to the profile's method which has access to private fields
        match self.profile.generate_model_data(&self.encoder) {
            Ok(()) => {
                log::info!("Model data generation completed successfully");
            }
            Err(e) => {
                log::error!("Failed to generate model data: {}", e);
            }
        }
    }
}

/// Entry point for model data generation
pub fn generate_model_data() {
    let generator = ModelDataGenerator::new();
    generator.generate();
}

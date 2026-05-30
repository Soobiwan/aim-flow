"""Generate restored local GenEval SPFC comparison outputs."""

from t2i_generation_common import generate_method


if __name__ == "__main__":
    generate_method(
        method="spfc",
        title="Restored SPFC",
        default_guidance_scale=4.5,
        archive_filename="geneval_spfc.zip",
    )

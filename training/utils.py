from dataclasses import dataclass


@dataclass
class ModulesNames:
    config = 'config'
    ae_model = 'ae_model'
    magnitude_encoder = 'magnitude_encoder'
    ref_net = 'ref_net'
    img_discriminator = 'img_discriminator'
    features_discriminator = 'features_discriminator'
    opt_magnitude_enc = 'opt_magnitude_encoder'
    opt_ref_net = 'opt_ref_net'
    opt_discriminators = 'opt_discriminators'
    opt_ae = 'opt_ae'

import torchio as tio

class VoxelUnitRandomElasticDeformation(tio.RandomElasticDeformation):
    """ Same as RandomElasticDeformation, but the max_displacement is in voxel units. """
    def apply_transform(self, subject: tio.data.Subject) -> tio.data.Subject:
        subject.check_consistent_spatial_shape()
        corrected_max_displacement = tuple(l * r for l, r in zip(self.max_displacement, subject.spacing))
        control_points = self.get_params(
            self.num_control_points,
            corrected_max_displacement,
            self.num_locked_borders,
        )

        arguments = {
            'control_points': control_points,
            'max_displacement': corrected_max_displacement,
            'image_interpolation': self.image_interpolation,
            'label_interpolation': self.label_interpolation,
        }

        transform = tio.ElasticDeformation(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

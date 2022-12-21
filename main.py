import math
import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray, Float, Shape
import typing


def main():
    # Set USE_TEX to False if no tex distribution is available on the operating device
    USE_TEX = True

    # Example 1: matching resolution: 1px, no disparity error: 0px
    ####################################################################################

    # Create a pinhole stereo camera model
    stereo_config = PinholeStereoCameraConfig(
        fov=math.radians(71.0),
        baseline=0.04,
        definition=640,
        matching_resolution=1,
        disparity_uncertainty=0,
    )
    stereo_model = PinholeStereoCameraModel(stereo_config)

    # Plot the stereo pair accuracy
    stereo_model.plot_depth_accuracy(USE_TEX)
    stereo_model.plot_min_obstacle_size(USE_TEX)

    # Example 2: subpixel matching resolution: 0.1px, typical disparity error: 1.5px
    ####################################################################################

    # Create a pinhole stereo camera model
    stereo_config = PinholeStereoCameraConfig(
        fov=math.radians(90.0),
        baseline=0.1,
        definition=1280,
        matching_resolution=0.1,
        disparity_uncertainty=1.5,
    )
    stereo_model = PinholeStereoCameraModel(stereo_config)

    # Plot the stereo pair accuracy
    stereo_model.plot_depth_accuracy(USE_TEX)
    stereo_model.plot_min_obstacle_size(USE_TEX)


########################################################################################


class PinholeStereoCameraConfig(typing.NamedTuple):
    """Pinhole stereo camera pair parameters.

    fov: field of view along the stereo axis [rad]
    baseline: distance between the optical center of the two cameras [m]
    definition: sensor definition along the stereo axis [pixels]
    matching_resolution: accuracy/resolution of the stereo matching (< 1 in case of
        sub-pixel matching accuracy) [pixel]
    disparity_uncertainty: typical disparity error (e.g. due stereo matching error or
        calibration error) [pixels]
    """

    fov: float
    baseline: float
    definition: int
    matching_resolution: float
    disparity_uncertainty: float


class PinholeStereoCameraModel:
    """Simple pinhole stereo camera model."""

    METER_TO_MILIMETER = 1000

    def __init__(self, config: PinholeStereoCameraConfig) -> None:
        self._config = config

        # camera focal length [pixels]
        self._focal_px = config.definition / (2.0 * math.tan(config.fov / 2.0))

    def plot_depth_accuracy(self, use_tex: bool = False) -> None:
        """Plot the stereo pair accuracy vs. depth."""
        POINT_COUNT = 10000

        # Generate data
        true_depth = self._generate_depth_vector(POINT_COUNT)

        continuous_depth_inf, continuous_depth_sup = self._continuous_depth_bounds(
            true_depth
        )

        discrete_depth = self._depth_to_discrete_depth(true_depth)
        discrete_depth_inf, discrete_depth_sup = self._discretized_depth_bounds(
            true_depth
        )

        discrete_depth_error = np.abs(true_depth - discrete_depth)
        max_discrete_depth_error = np.maximum(
            np.abs(true_depth - discrete_depth_inf),
            np.abs(true_depth - discrete_depth_sup),
        )

        # Create figure
        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)

        # Plot 1: Measured depth vs true depth
        ax1.plot(true_depth, true_depth, "--", color="black", linewidth=0.5)
        ax1.plot(
            true_depth,
            discrete_depth,
            color="royalblue",
            linewidth=1.75,
            label="No matching error",
        )
        ax1.plot(
            true_depth, continuous_depth_inf, ":", color="darkgrey", linewidth=0.75
        )
        ax1.plot(
            true_depth, continuous_depth_sup, ":", color="darkgrey", linewidth=0.75
        )

        ax1.fill_between(
            true_depth,
            discrete_depth_inf,
            discrete_depth_sup,
            color="forestgreen",
            linewidth=1,
            alpha=0.2,
            label="With matching error",
        )

        ax1.grid()
        ax1.set_xlim(left=0, right=np.max(discrete_depth))
        ax1.set_ylim(bottom=0, top=discrete_depth[-1] * 1.1)
        ax1.legend(loc="upper left")
        if use_tex:
            ax1.set_xlabel(r"true depth $d_{\mathrm{true}}$ (m)")
            ax1.set_ylabel(r"measured depth $d_{\mathrm{measured}}$ (m)")
        else:
            ax1.set_xlabel(f"true depth (m)")
            ax1.set_ylabel(f"measured depth (m)")

        # Plot 2: Measured depth error
        ax2.plot(
            true_depth,
            discrete_depth_error,
            color="royalblue",
            linewidth=1.75,
            label="No matching error",
        )
        ax2.fill_between(
            true_depth,
            discrete_depth_error,
            max_discrete_depth_error,
            color="forestgreen",
            alpha=0.2,
            label="With matching error",
        )

        ax2.grid()
        ax2.set_xlim(left=0, right=np.max(discrete_depth))
        ax2.set_ylim(bottom=0, top=discrete_depth[-1] * 1.1)
        ax2.legend(loc="upper left")
        if use_tex:
            ax2.set_xlabel(r"true depth $d_{\mathrm{true}}$ (m)")
            ax2.set_ylabel(r"measured depth error $\epsilon_{\mathrm{measured}}$ (m)")
        else:
            ax2.set_xlabel(f"true depth (m)")
            ax2.set_ylabel(f"measured depth error (m)")

        if use_tex:
            fig.suptitle(
                r"\textbf{Pinhole stereo depth accuracy} - FOV "
                f"{round(math.degrees(self._config.fov))}"
                f"°, definition "
                f"{self._config.definition}"
                f"px, baseline "
                f"{round(self._config.baseline * self.METER_TO_MILIMETER)}"
                f"mm, focal "
                f"{round(self._focal_px, 1)}"
                f"px, matching resolution "
                f"{self._config.matching_resolution}"
                f"px, disparity uncertainty "
                f"{self._config.disparity_uncertainty}"
                f"px"
            )
        else:
            fig.suptitle(
                f"Pinhole stereo depth accuracy - FOV"
                f"{round(math.degrees(self._config.fov))}"
                f"°, definition "
                f"{self._config.definition}"
                f"px, baseline "
                f"{round(self._config.baseline * self.METER_TO_MILIMETER)}"
                f"mm, focal "
                f"{round(self._focal_px, 1)}"
                f"px, matching resolution "
                f"{self._config.matching_resolution}"
                f"px, disparity uncertainty "
                f"{self._config.disparity_uncertainty}"
                f"px"
            )

        plt.show()

    def plot_min_obstacle_size(self, use_tex: bool = False) -> None:
        """Plot the min. detectable obstacle size on the optical center vs. depth."""
        POINT_COUNT = 1000

        angular_resolution = self._config.fov / self._config.definition
        depth = self._generate_depth_vector(POINT_COUNT)
        pixel_projected_size = depth * np.tan(angular_resolution)
        min_obstacle_size = 2 * pixel_projected_size

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(depth, min_obstacle_size)

        ax.grid()
        ax.set_xlim(left=0, right=np.max(depth))
        ax.set_ylim(bottom=0, top=np.max(min_obstacle_size))
        if use_tex:
            ax.set_title(
                r"\textbf{Approximate minimum detectable obstacle size at optical "
                r"center (Shanon criterion)} - FOV "
                f"{round(math.degrees(self._config.fov))}"
                f"°, definition "
                f"{self._config.definition}"
                f"px, baseline "
                f"{round(self._config.baseline * self.METER_TO_MILIMETER)}"
                f"mm, focal "
                f"{round(self._focal_px, 1)}"
                f"px, matching resolution "
                f"{self._config.matching_resolution}"
                f"px, disparity uncertainty "
                f"{self._config.disparity_uncertainty}"
                f"px"
            )
            ax.set_xlabel(r"depth $d$ (m)")
            ax.set_ylabel(r"smallest obstacle size $s_\mathrm{min}$ (m)")
        else:
            ax.set_xlabel(f"depth (m)")
            ax.set_ylabel(f"smallest obstacle size (m)")
            ax.set_title(
                f"Approximate minimum detectable obstacle size at optical center"
                f" (Shanon criterion) - FOV"
                f"{round(math.degrees(self._config.fov))}"
                f"°, definition "
                f"{self._config.definition}"
                f"px, baseline "
                f"{round(self._config.baseline * self.METER_TO_MILIMETER)}"
                f"mm, focal "
                f"{round(self._focal_px, 1)}"
                f"px, matching resolution "
                f"{self._config.matching_resolution}"
                f"px, disparity uncertainty "
                f"{self._config.disparity_uncertainty}"
                f"px"
            )

        plt.show()

    def _generate_depth_vector(self, point_count: int) -> NDArray[Shape["*"], Float]:
        min_depth = self._disparity_to_depth(self._config.definition)
        max_depth = self._disparity_to_depth(1.0)

        return np.linspace(min_depth, max_depth, point_count)

    def _depth_to_disparity(
        self, depth: float | NDArray[Shape["*"], Float]
    ) -> float | NDArray[Shape["*"], Float]:
        return self._focal_px * self._config.baseline / depth

    def _disparity_to_depth(
        self, disparity: float | NDArray[Shape["*"], Float]
    ) -> float | NDArray[Shape["*"], Float]:
        return self._focal_px * self._config.baseline / disparity

    def _depth_to_discrete_depth(self, depth):
        discrete_disparity = (
            np.round(self._depth_to_disparity(depth) / self._config.matching_resolution)
            * self._config.matching_resolution
        )

        return self._disparity_to_depth(discrete_disparity)

    def _continuous_depth_bounds(
        self, depth: NDArray[Shape["*"], Float]
    ) -> NDArray[Shape["*"], Float]:
        MIN_DISPARITY = 1e-6

        disparity = self._depth_to_disparity(depth)
        disparity_inf = np.maximum(
            np.abs(disparity - self._config.disparity_uncertainty), MIN_DISPARITY
        )
        disparity_sup = disparity + self._config.disparity_uncertainty

        return self._disparity_to_depth(disparity_inf), self._disparity_to_depth(
            disparity_sup
        )

    def _discretized_depth_bounds(
        self, depth: NDArray[Shape["*"], Float]
    ) -> NDArray[Shape["*"], Float]:
        MIN_DISPARITY = 1e-6

        discrete_disparity = (
            np.round(self._depth_to_disparity(depth) / self._config.matching_resolution)
            * self._config.matching_resolution
        )

        discrete_disparity_inf = np.minimum(
            np.maximum(
                np.abs(discrete_disparity - self._config.disparity_uncertainty),
                MIN_DISPARITY,
            ),
            self._config.definition,
        )
        discrete_disparity_sup = np.minimum(
            discrete_disparity + self._config.disparity_uncertainty,
            self._config.definition,
        )

        return self._disparity_to_depth(
            discrete_disparity_inf
        ), self._disparity_to_depth(discrete_disparity_sup)


if __name__ == "__main__":
    main()

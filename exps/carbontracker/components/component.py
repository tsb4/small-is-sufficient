import numpy as np

from carbontracker import exceptions
from carbontracker.components.gpu import nvidia
from carbontracker.components.cpu import intel
from carbontracker.components.apple_silicon.powermetrics import AppleSiliconCPU, AppleSiliconGPU

COMPONENTS = [
    {
        "name": "gpu",
        "error": exceptions.GPUError("No GPU(s) available."),
        "handlers": [nvidia.NvidiaGPU, AppleSiliconGPU],
    },
    {
        "name": "cpu",
        "error": exceptions.CPUError("No CPU(s) available."),
        "handlers": [intel.IntelCPU, AppleSiliconCPU],
    },
]


def component_names():
    return [comp["name"] for comp in COMPONENTS]


def error_by_name(name):
    for comp in COMPONENTS:
        if comp["name"] == name:
            return comp["error"]


def handlers_by_name(name):
    for comp in COMPONENTS:
        if comp["name"] == name:
            return comp["handlers"]


class Component:
    def __init__(self, name, pids, devices_by_pid):
        self.name = name
        if name not in component_names():
            raise exceptions.ComponentNameError(f"No component found with name '{self.name}'.")
        self._handler = self._determine_handler(pids=pids, devices_by_pid=devices_by_pid)
        self.power_usages = []
        self.cur_epoch = -1  # Sentry

    @property
    def handler(self):
        if self._handler is None:
            raise error_by_name(self.name)
        return self._handler

    def _determine_handler(self, pids, devices_by_pid):
        handlers = handlers_by_name(self.name)
        for h in handlers:
            handler = h(pids=pids, devices_by_pid=devices_by_pid)
            if handler.available():
                return handler
        return None

    def devices(self):
        return self.handler.devices()

    def available(self):
        return self._handler is not None

    def collect_power_usage(self, epoch):
        if epoch < 1:
            return

        if epoch != self.cur_epoch:
            self.cur_epoch = epoch
            # If we haven't measured for some epochs due to too slow
            # update_interval, we copy previous epoch measurements s.t.
            # there exists measurements for every epoch.
            diff = self.cur_epoch - len(self.power_usages) - 1
            if diff != 0:
                for _ in range(diff):
                    # Copy previous measurement lists.
                    latest_measurements = self.power_usages[-1] if self.power_usages else []
                    self.power_usages.append(latest_measurements)
            self.power_usages.append([])
        try:
            self.power_usages[-1].append(self.handler.power_usage())
            #print(self.name, self.power_usages, "collect_power_usages")
        except exceptions.IntelRaplPermissionError:
            # Only raise error if no measurements have been collected.
            if not self.power_usages[-1]:
                print(
                    "No sudo access to read Intel's RAPL measurements from the energy_uj file."
                    "\nSee issue: https://github.com/lfwa/carbontracker/issues/40"
                )
            # Append zero measurement to avoid further errors.
            self.power_usages.append([0])
        except exceptions.GPUPowerUsageRetrievalError:
            if not self.power_usages[-1]:
                print(
                    "GPU model does not support retrieval of power usages in NVML."
                    "\nSee issue: https://github.com/lfwa/carbontracker/issues/36"
                )
                # Append zero measurement to avoid further errors.
                self.power_usages.append([0])

    def energy_usage(self, epoch_times):
        """Returns energy (mWh) used by component per epoch."""
        energy_usages = []
        #print(self.name, self.power_usages)
        # We have to compute each epoch in a for loop since numpy cannot
        # handle lists of uneven length.
        for idx, (power, time) in enumerate(zip(self.power_usages, epoch_times)):
            # If no power measurement exists, try to use measurements from
            # later epochs.
            while not power and idx != len(self.power_usages) - 1:
                idx += 1
                power = self.power_usages[idx]
            if not power:
                power = [[0 for _ in range(len(self.devices()))]]
            avg_power_usage = np.mean(power, axis=0)
            #print(avg_power_usage, "avg", time)
            energy_usage = np.multiply(avg_power_usage, time)
            # Convert from J to kWh.
            for i in range(len(energy_usage)):
                if energy_usage[i] != 0:
                    energy_usage[i] /= 1#3600000 #Using Joules
            energy_usages.append(energy_usage)

        # Ensure energy_usages and epoch_times have same length by
        # copying latest measurement if it exists.
        diff = len(epoch_times) - len(energy_usages)
        if diff != 0:
            for _ in range(0, diff):
                # TODO: Warn that no measurements have been fetched.
                latest_energy = energy_usages[-1] if energy_usages else 0
                energy_usages.append(latest_energy)

        return energy_usages

    def init(self):
        self.handler.init()

    def shutdown(self):
        self.handler.shutdown()


def create_components(components, pids, devices_by_pid):
    components = components.strip().replace(" ", "").lower()
    if components == "all":
        return [Component(name=comp_name, pids=pids, devices_by_pid=devices_by_pid) for comp_name in component_names()]
    else:
        return [
            Component(name=comp_name, pids=pids, devices_by_pid=devices_by_pid) for comp_name in components.split(",")
        ]

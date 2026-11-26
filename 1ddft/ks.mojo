from std.math import pi
from std.python import Python
import std.math as math

import numojo as nm
from numojo.prelude import *
from numojo.core.memory.dlpack import from_dlpack


def main() raises:
    var np = Python.import_module("numpy")

    var n_grid: Int = 200
    var domain = nm.linspace[nm.f64](-5, 5, n_grid)
    var Deriv = -nm.eye[nm.f64](n_grid, n_grid) + nm.diagflat[nm.f64](nm.ones[nm.f64]([n_grid-1]),1)
    var grid_spacing = domain.item(1) - domain.item(0)
    Deriv /= grid_spacing

    var D2 = nm.dot(Deriv, -nm.transpose(Deriv))
    D2[Item(-1,-1)] = D2[Item(0,0)]

    var max_steps: Int = 1000
    var conv_thr: Float64 = 1e-5

    var density = nm.zeros[nm.f64]([n_grid])
    var ground_state_energy: List[Float64] = []

    var converged: Bool = False
    var num_iterations: Int = 0
    var num_electrons = 17
    var mixing_beta: Float64 = 1.0
    var omb: Float64 = 1.0 - mixing_beta

    var ext_pot: nm.NDArray[nm.f64] = domain**2

    var hartree_kernel = 1/nm.sqrt((
        nm.expand_dims(domain, 0) - nm.expand_dims(domain, 1)
             )**2 + 1e-1)

    var kinetic = -D2 / 2

    for i in range(max_steps):
        var ex_pot = exchange_pot(density)
        var hart_pot = hartree_pot(density, domain, hartree_kernel)
        
        var Hamiltonian = kinetic.copy()
        for k in range(n_grid):
            Hamiltonian[Item(k, k)] = (
                    Hamiltonian[Item(k,k)] + ex_pot.item(k) + ext_pot.item(k) + hart_pot.item(k)
                    )


        # There is not currently an implementation of a complex eigenvalue algorithm in 
        # the pure-mojo libraries
        #----------------------------------------------
        var diagonalized = np.linalg.eigh(Hamiltonian.to_numpy())

        var ei_np = np.ascontiguousarray(diagonalized[0])
        var psi_np = np.ascontiguousarray(diagonalized[1])

        var ei: nm.NDArray[nm.f64] = nm.array[nm.f64](ei_np)
        var psi: nm.NDArray[nm.f64] = nm.array[nm.f64](psi_np)
        # --------------------------------------------
        ground_state_energy.append(ei.item(0))

        # density = (1.0 - mixing_beta) * density + mixing_beta * new_density
        var new_density = compute_density(psi, domain, num_electrons)
        new_density *= mixing_beta
        density *= omb 
        density += new_density

        if i == 0:
            print("Initial state computed")
            continue

        num_iterations += 1

        var energy_diff : Float64 = math.abs(ground_state_energy[i] - ground_state_energy[i-1])
        print(t"iteration: {num_iterations}, energy: {ground_state_energy[i]}, estimated accuracy: {energy_diff}")

        if energy_diff < conv_thr:
            converged = True
            break 

    if converged:
        print(t"Convergence achieved in {num_iterations} iterations")
    else:
        print("system not converged, too many iterations")


def compute_density(psi: nm.NDArray[nm.f64], domain: nm.NDArray[nm.f64], num_electrons: Int) raises -> nm.NDArray[nm.f64]:

    var num_levels: Int = num_electrons // 2

    var occs: List[Float64] = List[Float64](capacity=num_levels+1)
    for _ in range(num_levels):
        occs.append(2.0) 

    if num_electrons % 2 == 1:
        num_levels += 1
        occs.append(1.0)

    var density: nm.NDArray[nm.f64] = nm.zeros[nm.f64]([domain.size])
    var tpsi = nm.transpose(psi)

    for i in range(num_levels):
        var occ = occs[i]
        var psi_i = tpsi[i]
        var sq = psi_i**2
        var I: Float64 = nm.trapezoid(psi_i**2, domain)
        sq *= Float64(occ)/I
        density += sq

    return density^

def exchange_pot(density: nm.NDArray[nm.f64]) raises -> nm.NDArray[nm.f64]:
    comptime cbrt: Float64 = 1.0/3.0
    comptime c: Float64 = -(3/pi)**cbrt
    var pot: nm.NDArray[nm.f64] = density**cbrt
    pot *= c
    return pot^


def hartree_pot(density: nm.NDArray[nm.f64], domain: nm.NDArray[nm.f64], kernel: nm.NDArray[nm.f64]) raises -> nm.NDArray[nm.f64]:
    var h = domain.item(1) - domain.item(0)
    var n: Int = domain.size
    var pot: nm.NDArray[nm.f64] = nm.zeros[nm.f64]([n])

    for i in range(n):
        var row = kernel[i].copy()
        row *= density
        row *= h
        pot[Item(i)] = nm.sum(row)
    return pot^

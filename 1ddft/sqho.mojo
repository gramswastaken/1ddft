from std.python import Python, PythonObject
import numojo as nm

def main() raises:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    var n_grid: Int = 200
    var x: nm.NDArray[nm.f64] = nm.linspace[nm.f64](-5, 5, n_grid)
    var y: nm.NDArray[nm.f64] = nm.sin(x)

    # derivatives
    var h = x[1].item() - x[0].item()
    var D = -nm.eye[nm.f64](n_grid, n_grid) + nm.diagflat(nm.ones[nm.f64]([n_grid-1]), 1)
    D = D / h

    var D2 = nm.dot(D, -nm.transpose(D))

    # D2[0,0] as dtype
    ref es: Float64 = D2[0,0].item()
    ref ee: Float64 = D2[-1,-1].item()
    ee = es

    var sinx = nm.dot(D2, y)

    var npx = x.to_numpy()
    var npD2 = D2.to_numpy()
    
    # harmonic oscillator
    var pot = nm.diagflat(x*x).to_numpy()

    var result = np.linalg.eigh((-npD2 / 2) + pot)
    var eig_harm = result[0]
    var psi_harm = result[1]

    var num_eig = len(eig_harm)

    var fig_ax = plt.subplots()
    var fig = fig_ax[0]
    var ax = fig_ax[1]

    for i in range(5):
        var psi_i = np.take(psi_harm, i, axis=1)
        ax.plot(npx,psi_i,label=String(eig_harm[i])) 

    plt.show()


def plot_da_data(ax: PythonObject, x: nm.NDArray[nm.f64], y: nm.NDArray[nm.f64]) raises:
    var plt = Python.import_module("matplotlib.pyplot")
    plt.figure()
    plt.plot(x.to_numpy()[1:-1], y.to_numpy()[1:-1])
    plt.show()


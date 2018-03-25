import numpy as np
import matplotlib
import matplotlib.ticker


def setup_log_linear_axes(axes, **kwargs):
    axes.set_yscale('log')
    axes.set_xscale('linear')
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else r"q ($\mathregular{Å^{-1}}$)")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else "I(q)")
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def setup_log_log_axes(axes, **kwargs):
    axes.set_yscale('log')
    axes.set_xscale('log')
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else r"q ($\mathregular{Å^{-1}}$)")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else "I(q)")
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def plot_saxs(data, axes, errorbars=True, **kwargs):
    if 'label' not in kwargs and hasattr(data, 'name'):
        kwargs['label'] = data.name

    if errorbars:
        axes.errorbar(data.q, data.i, yerr=data.error, **kwargs)
    else:
        axes.plot(data.q, data.i, **kwargs)


def setup_gunier_axes(axes, **kwargs):
    axes.set_xscale('linear')
    axes.set_yscale('log', basey=np.e)
    axes.get_yaxis().set_major_locator(matplotlib.ticker.LinearLocator())
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else r"$\mathregular{q^2}$ ($\mathregular{Å^{-2}}$)")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else 'I(q)')
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def setup_kratky_axes(axes, **kwargs):
    axes.set_yscale('linear')
    axes.set_xscale('linear')
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else r"q ($\mathregular{Å^{-1}}$)")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else r"I(q) * $\mathregular{q^2}$")
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def plot_kratky(data, axes, errorbars=True, **kwargs):
    label = ''
    if 'label' in kwargs:
        label = kwargs['label']
    elif hasattr(data, 'name'):
        label = data.name

    x = data.q
    y = data.i * data.q * data.q
    error = data.error * data.q * data.q if data.error is not None else None
    if errorbars:
        axes.errorbar(x, y, yerr=error, label=label, **kwargs)
    else:
        axes.plot(x, y, label=label, **kwargs)


def setup_normalized_kratky_axes(axes, **kwargs):
    axes.set_yscale('linear')
    axes.set_xscale('linear')
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else "q * Rg")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else r"I(q) * $\mathregular{(q * Rg)^2}$")
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def plot_normalized_kratky(data, axes, errorbars=True, **kwargs):
    if 'rg' in kwargs:
        rg = kwargs.pop('rg')
    elif hasattr(data, 'rg'):
        rg = data.rg
    else:
        raise AttributeError("Either provide the radius of gyration as an argument or set it as a property of the "
                             "input Saxscurve")

    label = ''
    if 'label' in kwargs:
        label = kwargs.pop('label')
    elif hasattr(data, 'name'):
        label = data.name

    x = data.q * rg
    y = data.i * data.q * data.q * rg * rg
    error = data.error * data.q * data.q if data.error is not None else None
    if errorbars:
        axes.errorbar(x, y, yerr=error, label=label, **kwargs)
    else:
        axes.plot(x, y, label=label, **kwargs)


def setup_pr_axes(axes, **kwargs):
    axes.set_yscale('linear')
    axes.set_xscale('linear')
    axes.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else "r (Å)")
    axes.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else "P(r)")
    if 'title' in kwargs:
        axes.set_title(kwargs['title'])


def plot_pr(data, axes, **kwargs):
    if 'label' not in kwargs and hasattr(data, 'name'):
        kwargs['label'] = data.name

    axes.plot(data.distances, data.pvals, **kwargs)


def setup_3d_plot(figure, **kwargs):
    return figure.add_subplot(1, 1, 1, projection='3d', **kwargs)


def plot_3d(datas, axes, zvals=None, **kwargs):
    for i, data in enumerate(datas):
        if zvals is not None:
            axes.plot(data.q, data.i, zs=zvals[i], zdir='y', **kwargs)
        else:
            axes.plot(data.q, data.i, zs=i, zdir='y', **kwargs)
    axes.autoscale()


def fancy_gunier(saxsdata, axes, rg, i0, first, last, color='red'):
    # the argument `last` is inclusive but it will be used to slice the data
    last += 1
    # x-axis data
    q_squared = saxsdata.q * saxsdata.q
    # set up axes -- the Y axes should be natural log scaled, but display linear values because they look nicer
    # (matplotlib is particularly heinous at ticking and labeling log scale axes that are not base 10)
    axes.set_xscale('linear')
    axes.set_yscale('log', basey=np.e)
    axes.get_yaxis().set_major_locator(matplotlib.ticker.LinearLocator())
    axes.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes.set_xlabel(r"$\mathregular{q^2}$ ($\mathregular{Å^{-2}}$)")
    axes.set_ylabel('I(q)')
    # plot the curve as a whole
    axes.scatter(x=q_squared, y=saxsdata.i, c="gray", label="Data")
    # plot the gunier points
    axes.scatter(x=q_squared[first:last], y=saxsdata.i[first:last], c="black", label="Gunier region")
    # plot the fit line implied by rg and i0
    gunier_fit_y = np.exp(np.log(i0) - q_squared[first:last] * rg * rg / 3.0)
    ss_averages = np.sum(np.square(np.log(saxsdata.i[first:last]) - np.mean(np.log(saxsdata.i[first:last]))))
    ss_residuals = np.sum(np.square(np.log(saxsdata.i[first:last]) - np.log(gunier_fit_y)))
    rsquared = 1.0 - ss_residuals/ss_averages
    axes.plot([q_squared[first], q_squared[last - 1]], [gunier_fit_y[0], gunier_fit_y[-1]], marker='', c=color,
              label="Gunier fit,\nRg = {:.1f} $\\mathregular{{R^2}}$ = {:.3f}".format(rg, rsquared))
    # fix x axis limits if < 0.0
    if axes.get_xlim()[0] < 0.0:
        axes.set_xlim(left=0.0)

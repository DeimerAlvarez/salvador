"""Plots of each line: PlotManager (and NullPlotter for runs that save no plot), plot style."""
import math
from pathlib import Path
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from .config import cfg, plt

class NullPlotter:
    """SPEED: stand-in for PlotManager in automatic runs whose plot is never saved
    (parameter-error reruns). Before, those runs drew a full figure at every iteration
    and never closed it."""
    def update(self, plot_data):
        """Nothing to draw."""
        pass
    def save(self, path_tmp, line):
        """Nothing to save."""
        return None
    def write_log(self, path_plot_file, filename):
        """Nothing to record."""
        pass
    def close(self):
        """Nothing to close."""
        pass


def apply_plot_style(ax, xlabel, ylabel):
    """
    Apply the requested aesthetic settings to a given axis (ax).
    """
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    # Ticks
    ax.tick_params(which='major', direction='in', length=8, top=True, right=True, labelsize=14)
    ax.tick_params(which='minor', direction='in', length=4, top=True, right=True)
    ax.minorticks_on()

    # Axes (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)


def format_axes(ax):
    """Style of an axis: ticks inside on all sides, minor ticks, thick frame."""
    ax.tick_params(which='major', direction='in', length=8, top=True, right=True, labelsize=12)
    ax.tick_params(which='minor', direction='in', length=4, top=True, right=True)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)

class PlotManager:
    """Plot of a line, created once and updated after every fit: spectrum and synthetics (left),
    zoom on the line (top right), chi2 against abundance, or against rotation, for ROTAT (bottom right)."""
    def __init__(self, star_title, tana='ABU'):
        """creates the elements for the plot once"""

        self.tana = tana
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6.75))

        # --- The 3 panels ---
        self.ax1 = self.fig.add_axes([0.075, 0.1, 0.52, 0.85])
        self.ax2 = self.fig.add_axes([0.68, 0.1, 0.3, 0.3])
        self.ax3 = self.fig.add_axes([0.68, 0.5, 0.3, 0.45])

        # --- Fixed settings of the main plot (ax1) ---
        self.ax1.set_xlabel(r'Wavelength [$\AA$]', fontsize=15)
        self.ax1.set_ylabel('Normalized flux', fontsize=15)
        self.ax1.set_title(f'{star_title}', fontsize=12)
        format_axes(self.ax1)
        self.ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Lines created with empty data, filled by update()
        self.line_syn1, = self.ax1.plot([], [], c='limegreen', zorder=2)
        self.line_syn2, = self.ax1.plot([], [], c='red', zorder=2)
        self.line_bestfit, = self.ax1.plot([], [], c='blue', zorder=3)
        if cfg.plot_type == 0:
            self.line_spec, = self.ax1.plot([], [], c='k', zorder=1)
        elif cfg.plot_type == 1:
            self.line_spec, = self.ax1.plot([], [], 'ko', markersize=4, zorder=1)
        elif cfg.plot_type == 2:
            self.line_spec, = self.ax1.step([], [], c='k', zorder=1)

        self.scatter_cont_reg = self.ax1.scatter([], [], marker='x', c='r', zorder=5, s=10)
        self.vline1 = self.ax1.axvline(0, c='k', ls='--')
        self.vline2 = self.ax1.axvline(0, c='k', ls='--')
        self.fill_regions = []
        self.vline_new1 = self.ax1.axvline(0, c='k', ls='--', visible=False)
        self.vline_new2 = self.ax1.axvline(0, c='k', ls='--', visible=False)

        # --- Fixed settings of ax2, depending on 'tana' ---
        format_axes(self.ax2)
        if self.tana in ['ABU', 'ABU2', 'ABUF', 'ABU_GF', 'ISOR']:
            if self.tana == 'ISOR':
                self.ax2.set_xlabel('Isotopic ratio', fontsize=15)
            else:
                self.ax2.set_xlabel('Abu', fontsize=15)
            self.ax2.set_ylabel(r'$\chi^2$', fontsize=15)
            # EMPTY scatter elements
            self.scatter_chi_curva = self.ax2.scatter([], [], c='r', s=30, ec='k', lw=0.5)
            self.scatter_chi_minimo = self.ax2.scatter([], [], c='b', s=40, ec='k', lw=0.5, zorder=3)
        elif self.tana == 'ROTAT':
            self.ax2.set_xlabel('Rotational Velocity [km/s]', fontsize=15)
            self.ax2.set_ylabel('Abundance [dex]', fontsize=15)
            # For the contour plot only the marker of the minimum is created:
            # the contours are redrawn at every update.
            self.scatter_minimo_rotat = self.ax2.scatter([], [], marker='+', color='red', s=80)
            self.colorbar = None # No colorbar at the start

        # --- Fixed settings of the chi-square plot (ax2) ---
        self.ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        # --- EMPTY scatter elements: one for the curve, one for the minimum ---
        self.scatter_chi_curva = self.ax2.scatter([], [], c='r', s=30, ec='k', lw=0.5)
        self.scatter_chi_minimo = self.ax2.scatter([], [], c='b', s=40, ec='k', lw=0.5, zorder=3, label='_nolegend_')

        # --- Fixed settings of the zoom plot (ax3) ---
        self.ax3.set_xlabel(r'Wavelength [$\AA$]', fontsize=15)
        self.ax3.set_ylabel('Norm flux', fontsize=15)
        format_axes(self.ax3)
        self.ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.line_zoom_fit, = self.ax3.plot([], [], c='blue', zorder=2)
        self.line_zoom_spec, = self.ax3.plot([], [], c='k', zorder=1)
        self.vline_zoom = self.ax3.axvline(0, c='k', ls='--', label='_nolegend_') # Etichetta temporanea

        # SPEED: render on screen only in interactive mode. In automatic mode savefig() renders
        # the figure anyway (saved PDFs are pixel-identical); draw + pause cost ~0.35 s per figure.
        if cfg.inter == 'y':
            self.fig.canvas.draw()
            plt.pause(0.1)


    def update(self, plot_data):
        """Redraw the plot with the data of the last fit (dictionary plot_data from the fit)."""
        wl_syn = plot_data['wl_syn']
        fl_syn1 = plot_data['fl_syn1']
        fl_syn2 = plot_data['fl_syn2']
        xfit = plot_data['xfit']
        bestfit = plot_data['bestfit']
        xspec = plot_data['xspec']
        yspec = plot_data['yspec']
        fit1 = plot_data['fit1']
        fit2 = plot_data['fit2']
        line = plot_data['line']
        window = plot_data['window']
        type_norm = plot_data['type_norm']
        wv_left = plot_data['wv_left']
        wv_right = plot_data['wv_right']
        elem = plot_data['elem']
        #
        fit1_new = plot_data.get('fit1_new')
        fit2_new = plot_data.get('fit2_new')


        """Update the plot"""
        # --- Update of the main plot (ax1) ---
        self.line_syn1.set_data(wl_syn, fl_syn1)
        self.line_syn2.set_data(wl_syn, fl_syn2)
        self.line_bestfit.set_data(xfit, bestfit)
        self.line_spec.set_data(xspec, yspec)

        if fit1_new is not None and fit2_new is not None:
            # If present: update the data and show the lines
            self.vline_new1.set_xdata([fit1_new, fit1_new])
            self.vline_new2.set_xdata([fit2_new, fit2_new])
            self.vline_new1.set_visible(True)
            self.vline_new2.set_visible(True)
        else:
            # If not present: hide the lines
            self.vline_new1.set_visible(False)
            self.vline_new2.set_visible(False)

        self.vline1.set_xdata([fit1, fit1])
        self.vline2.set_xdata([fit2, fit2])

        # Elements such as fill_between and text are easier to remove and create again
        for fill in self.fill_regions:
                fill.remove()
        self.fill_regions = []
        for txt in self.ax1.texts:
            txt.remove()


        # Update limits and ticks
        mask_line = (np.array(wl_syn) > line - window / 2) & (np.array(wl_syn) < line + window / 2)
        B = np.array(fl_syn2[mask_line])
        X0 = line - (window / 2)
        X1 = line + (window / 2)
        xticks = np.array(range(int(math.ceil(X0)), int(math.floor(X1)) +1))

        if len(xticks) >= 10:
            step_tick = 2
        else:
            step_tick = 1
        xticks = np.arange(xticks[0], xticks[-1]+1, step_tick)

        if type_norm == 2:
            for l, (wleft, wright) in enumerate(zip(wv_left, wv_right)):
                fill_obj = self.ax1.fill_between([wleft, wright], [0.0, 0.0], [1.2, 1.2], color='yellow', alpha=0.9, zorder=0)
                self.fill_regions.append(fill_obj)

                if wleft > xticks[0] - 0.4 and wright < xticks[-1] + 0.4:
                    self.ax1.text((wleft + wright) / 2, 1.1, l + 1, ha='center', weight='bold')
                elif wleft < xticks[0] and wright > xticks[0]:
                    self.ax1.text(wright - 0.1, 1.1, l + 1, ha='center', weight='bold')
                elif wright > xticks[-1] and wleft < xticks[-1]:
                    self.ax1.text(wleft + 0.1, 1.1, l + 1, ha='center', weight='bold')


        self.ax1.set_xlim(X0, X1)
        self.ax1.set_ylim(min(B) - 0.1, 1.2)
        self.ax1.set_xticks(xticks)

        # --- Update of the second panel (ax2), depending on the mode ---
        if self.tana.startswith(('ABU', 'ABU2', 'MOLEC', 'ABU_GF', 'ABUF', 'ISOR')):
            self._update_ax2_abundance(plot_data)
        elif self.tana == 'ROTAT':
            self._update_ax2_rotation(plot_data)

        # --- Update of the zoom plot (ax3) ---
        xspec_vals = xspec.value if hasattr(xspec, 'value') else xspec
        yspec = yspec.value if hasattr(yspec, 'value') else yspec

        mask_reg1_fit = (xfit > fit1) & (xfit < fit2)
        mask_reg1_spec = (xspec_vals > fit1) & (xspec_vals < fit2)
        if fit1_new is not None:
            mask_reg2_fit = (xfit > fit1_new) & (xfit < fit2_new)
            mask_fit = mask_reg1_fit | mask_reg2_fit
            mask_reg2_spec = (xspec_vals > fit1_new) & (xspec_vals < fit2_new)
            mask_spec = mask_reg1_spec | mask_reg2_spec
        else:
            mask_fit = mask_reg1_fit
            mask_spec = mask_reg1_spec

        xfit_cut, bestfit_cut = np.where(mask_fit, xfit, np.nan), np.where(mask_fit, bestfit, np.nan)
        xspec_cut, yspec_cut = np.where(mask_spec, xspec, np.nan), np.where(mask_spec, yspec, np.nan)

        # Update the lines of the zoom plot
        self.line_zoom_fit.set_data(xfit_cut, bestfit_cut)
        self.line_zoom_spec.set_data(xspec_cut, yspec_cut)

        # Update the position of the vertical line and its label for the legend
        self.vline_zoom.set_xdata([line, line]) # The position is passed as a list
        self.vline_zoom.set_label(f'{elem} {line:.3f}')
        self.ax3.legend(loc='best', frameon=True, edgecolor='k')

        # 4. Update the limits and ticks of the x axis
        def def_ticks(ff1, ff2):
            """Three ticks (multiples of 0.1, inside the interval ff1-ff2): first, middle, last."""
            tick_min = np.round(np.ceil(ff1 / 0.1) * 0.1, 2)
            tick_max = np.round(np.floor(ff2 / 0.1) * 0.1, 2)
            if tick_min <= ff1:
                tick_min = round(tick_min + 0.1, 2)
            if tick_max >= ff2:
                tick_max = round(tick_max - 0.1, 2)
            tick_mid = np.round((tick_min + tick_max) / 2.0, 2)
            return [tick_min, tick_mid, tick_max]

        if fit1_new is not None:
            self.ax3.set_xlim(fit1 - 0.01, fit2_new + 0.01)
            xticks = def_ticks(fit1, fit2_new)
            self.ax3.set_xticks(xticks)
        else:
            self.ax3.set_xlim(fit1 - 0.01, fit2 + 0.01)
            xticks = def_ticks(fit1, fit2)
            self.ax3.set_xticks(xticks)

        overall_min = np.nanmin(np.concatenate((yspec_cut, bestfit_cut)))
        overall_max = np.nanmax(np.concatenate((yspec_cut, bestfit_cut)))
        self.ax3.set_ylim(overall_min - 0.02, overall_max + 0.02)

        # Draw the changes
        # SPEED: only in interactive mode (~90 ms per update otherwise)
        if cfg.inter == 'y':
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _update_ax2_abundance(self, data):
        """Bottom-right panel for abundances: chi2 against abundance near the minimum."""
        abund, chi, abu_fin, elem = data['abund'], data['chi'], data['abu_fin'], data['elem']
        if data['type_abu'] != 'ISOR':
            mask_abu = (abund > np.mean(abund) - 0.2) & (abund < np.mean(abund) + 0.2)
        elif data['type_abu'] == 'ISOR':
            mask_abu = (abund > np.mean(abund) - 5) & (abund < np.mean(abund) + 5)

        abund_cut, chi_cut = abund[mask_abu], chi[mask_abu]

        padding = (np.max(chi_cut) - np.min(chi_cut)) * 0.10

        self.scatter_chi_curva.set_offsets(np.c_[abund_cut, chi_cut])
        # the point on the curve at the abundance found (with CALC_BEST = n it is the minimum of the curve)
        self.scatter_chi_minimo.set_offsets(np.c_[[abu_fin], [chi[np.argmin(np.abs(abund - abu_fin))]]])
        if data['type_abu'] != 'ISOR':
            self.scatter_chi_minimo.set_label(f'A({elem})={abu_fin:.2f}')
        elif data['type_abu'] == 'ISOR':
            self.scatter_chi_minimo.set_label(f'ISOR ({elem})={abu_fin:.0f}')
        self.ax2.legend(loc='best', frameon=True, edgecolor='k')
        self.ax2.set_xlim(abund_cut[0], abund_cut[-1])
        self.ax2.set_ylim(min(chi_cut) - padding, max(chi_cut) + padding)

    def _update_ax2_rotation(self, data):
        """Bottom-right panel for ROTAT: contour plot of chi2 (abundance x rotation), or chi2 against
        rotation if the abundance is fixed; best fit with the Monte Carlo errors."""
        # Data needed for this plot
        rot_vals = np.asarray(data['finer_rotations'])
        abu_vals = np.asarray(data['finer_abundances'])
        chi_grid = np.atleast_2d(data['min_chi2_fine']) # chi-square grid
        best_rot = data['best_rot_fine']
        best_abu = data['best_abu_fine']
        # Monte Carlo errors (shown in the legend)
        err_rot = data.get('err_rot')
        err_abu = data.get('err_abu')
        rot_txt = f'{best_rot:.0f}' + (f' $\\pm$ {np.floor(err_rot + 0.5):.0f}' if err_rot is not None else '') + ' km/s'

        # Clear the axis before drawing again
        self.ax2.clear()
        format_axes(self.ax2) # Apply the formatting again
        self.ax2.set_xlabel('Rotational Velocity [km/s]', fontsize=12)

        # More than one abundance?
        if len(abu_vals) > 1:
            # ==========================================
            # 2D CASE: contour plot (abundance free)
            # ==========================================
            self.ax2.set_ylabel('Abundance [dex]', fontsize=12)
            self.ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Logarithmically spaced levels: the region of the minimum is better resolved (with linear
            # levels the large values far from the minimum squeeze the valley into a few colours)
            chi_min, chi_max = np.nanmin(chi_grid), np.nanmax(chi_grid)
            if chi_min > 0 and chi_max > chi_min:
                levels = np.geomspace(chi_min, chi_max, 21)
            else:
                levels = 20

            # New contours
            self.ax2.contourf(rot_vals, abu_vals, chi_grid, levels=levels, cmap='viridis')
            self.ax2.contour(rot_vals, abu_vals, chi_grid, levels=levels, colors='white', linewidths=0.5)

            # Minimum, with the Monte Carlo error bars
            abu_txt = f'{best_abu:.2f}' + (f' $\\pm$ {err_abu:.2f}' if err_abu is not None else '')
            self.ax2.errorbar(best_rot, best_abu, xerr=err_rot or 0, yerr=err_abu or 0, fmt='+', color='red',
                    ms=10, mew=1.5, elinewidth=1, capsize=2, zorder=5, label=f'rot = {rot_txt}\nA = {abu_txt}')

        else:
            # ==========================================
            # 1D CASE: line plot (abundance fixed)
            # ==========================================
            self.ax2.set_ylabel(r'$\chi^2$', fontsize=12)

            # chi_grid has shape (1, len(rot_vals)): take the 1D array
            chi_1d = chi_grid[0, :]

            # chi-square as a function of the rotation
            self.ax2.scatter(rot_vals, chi_1d, color='r', ec='k', lw=0.5, s=30)

            # Minimum of the curve (nanargmin: robust to NaN)
            min_chi2_val = chi_1d[np.nanargmin(chi_1d)]
            self.ax2.scatter(best_rot, min_chi2_val, color='b', s=40, zorder=5, ec='k', lw=0.5,
                             label=f'rot = {rot_txt}\nA = {best_abu:.2f} (fixed)')

            # Dashed vertical line and Monte Carlo error band
            self.ax2.axvline(best_rot, color='red', linestyle='--', alpha=0.5)
            if err_rot:
                self.ax2.axvspan(best_rot - err_rot, best_rot + err_rot, color='red', alpha=0.12, lw=0)

        # Legend in both cases
        self.ax2.legend(loc='best', frameon=True, edgecolor='k', fontsize=9)

    def save(self, path_tmp, line):
        """Save the plot as line_<wavelength>.pdf in path_tmp. Returns the file name."""
        self.filename = f'line_{line}.pdf'
        filepath = Path(path_tmp) / self.filename
        self.fig.savefig(filepath)
        return self.filename

    def write_log(self, path_plot_file, filename):
        """Record the name of the saved plot, for the final PDF with all the lines."""
        with open(path_plot_file, 'a') as text_file:
            text_file.write(filename + "\n")

    def close(self):
        """Close the plot window."""
        plt.ioff()
        plt.close(self.fig)

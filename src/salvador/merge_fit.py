"""salvador-merge: merge the .fit files of the same star (e.g. from different settings or
spectral ranges) and compute its final abundances again, optionally with an external [Fe/H].

The input file lists the .fit files of each star, then the name of the merged file (optionally
followed by [Fe/H]); stars are separated by '###' (see docs/example_merge_fit.txt).
Results: the merged .fit files, their 3-panel Fe plots and abundance_final_merged.txt.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, mad_std
from scipy import stats

from .elements import solar_table_path


def read_solar_data(filename):
    try:
        data = []
        with open(solar_table_path(), 'r') as file: # table installed with salvador
            for line in file:
                if not line.startswith('#'):
                    parts = line.strip().split()
                    if parts:
                        data.append(parts)
        return data
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        return None

def abu_dict(solar_data, column=2):
    abundance_dict = {}
    for row in solar_data:
        if len(row) >= column + 1:  # Verifica che la colonna richiesta esista
            element_name = row[1]
            try:
                abundance = float(row[column])
                abundance_dict[element_name] = abundance
            except ValueError:
                print(f"Warning: Skipping invalid abundance in row: {row}")
        else:
            print(f"Warning: Skipping row with insufficient columns: {row}")
    return abundance_dict
    

def get_element_symbol(nelem, symbol_map):
    int_nelem = int(nelem)
    symbol = symbol_map.get(int_nelem)
    if symbol:
        if "{:.2f}".format(nelem).endswith("00"):
            return symbol
        elif "{:.2f}".format(nelem).endswith("01"):
            return f"{symbol}II"
        else:
            return f"{symbol}III"
    else:
        return None
        
        
def extract_abu_fe(nelem, abu, abu_mc, nel, flag_abu, abu_pT=None, abu_pg=None, abu_pv=None):
    # Determina la modalità di calcolo in base agli input
    full_error_mode = all(err is not None for err in [abu_pT, abu_pg, abu_pv])
        
    # 1. Filtra i dati (logica identica per entrambe le modalità)
    if calc_flags == 0:
        mask = (nelem == nel) & (abu != 99.99) & np.isin(flag_abu, [0, 1])
    elif calc_flags == 1:
        mask = (nelem == nel) & (abu != 99.99) & np.isin(flag_abu, [1])
    
    nlin = np.count_nonzero(mask)
        
    original_indices = np.where(mask)[0]
    
    if nlin == 0:
        return (0, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, flag_abu, 99.99) if full_error_mode else (0, 99.99, 99.99, flag_abu, 99.99)
        
    abu_elem = abu[mask]
    mc_err_elem = abu_mc[mask]
        
    # 2. Ottieni l'abbondanza solare
    elem_symbol = get_element_symbol(nel, symbol_map).replace('II', '').replace('III', '')
    abu_sol = abundance_link.get(elem_symbol)
    
    # 3. Calcola media e componenti di errore in base al numero di righe
    s1, s2, s3, s4 = 99.99, 99.99, 99.99, 99.99
    
    if nlin == 1:   
        mean = abu_elem[0] - abu_sol
        s1 = mc_err_elem[0]
        flag_abu[original_indices[0]] = 1
        if full_error_mode:
            s2, s3, s4 = abu_pT[mask][0], abu_pg[mask][0], abu_pv[mask][0]
            
    else: # nlin > 1
        clip_mask = np.ones(nlin, dtype=bool) # Maschera per lo clipping
        
        if calc_flags == 0:
            if nlin >= 3:
                disp = mad_std(abu_elem)
                if np.isfinite(disp) and disp > 0.0:
                    clipped_abu = sigma_clip(abu_elem, sigma = sclip, cenfunc='median', stdfunc=mad_std)
                    if clipped_abu.count() > 0:
                        clip_mask = ~clipped_abu.mask
                        nlin = clipped_abu.count() # Aggiorna nlin effettivo
                    
            final_indices_used = original_indices[clip_mask]
            flag_abu[final_indices_used] = 1 # Imposta i flag per tutte le righe usate

            final_indices_rejected = original_indices[~clip_mask]
            flag_abu[final_indices_rejected] = 0
            
        mean = np.mean(abu_elem[clip_mask]) - abu_sol
        s1 = np.std(abu_elem[clip_mask], ddof=1) / np.sqrt(nlin)
                
        if full_error_mode:
            s2 = np.mean(abu_pT[mask][clip_mask])
            s3 = np.mean(abu_pg[mask][clip_mask])
            s4 = np.mean(abu_pv[mask][clip_mask])
    
    # 4. Controllo finale per NaN
    if np.isnan(mean):
        return (0, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, flag_abu, 99.99) if full_error_mode else (0, 99.99, 99.99, flag_abu, 99.99)
        
    # 5. Restituisci il risultato formattato in base alla modalità
    if full_error_mode:
        sigma_total = np.sqrt(s1**2 + s2**2 + s3**2 + s4**2)
        sigma_params = np.sqrt(s2**2 + s3**2 + s4**2)
        return (nlin, np.round(mean, 2), np.round(sigma_total, 2), np.round(s1, 2), np.round(s2, 2), np.round(s3, 2), np.round(s4, 2), flag_abu, np.round(sigma_params, 2))
    else:
        return nlin, np.round(mean, 2), np.round(s1, 2), flag_abu, 99.99


def extract_abu(nelem, abu, abu_mc, nel, feh, flag_abu, s_stat_Fe, param_errors=None, fe_errors=None):
    full_error_mode = param_errors is not None and fe_errors is not None

    # 1. Filtra i dati per l'elemento X
    if calc_flags == 0:
        mask = (nelem == nel) & (abu != 99.99) & np.isin(flag_abu, [0, 1])
    elif calc_flags == 1:
        mask = (nelem == nel) & (abu != 99.99) & np.isin(flag_abu, [1])
    
    nlin = np.count_nonzero(mask)
    
    original_indices_X = np.where(mask)[0]
        
    if nlin == 0:
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}
        return 0, 99.99, 99.99, flag_abu, 99.99, 99.99, params_XH
        
    # Applica la maschera ai dati obbligatori
    abu_elem = abu[mask]
    mc_err_elem = abu_mc[mask]
        
    # 2. Ottieni l'abbondanza solare di riferimento
    elem_symbol = get_element_symbol(nel, symbol_map).replace('II', '').replace('III', '')
    abu_sol = abundance_link.get(elem_symbol)
    
    # 3. Calcola la media [X/Fe] e le componenti di errore per X
    sigma_T_X, sigma_g_X, sigma_v_X = 0.0, 0.0, 0.0
    
    if nlin == 1:
        mean = abu_elem[0] - abu_sol - feh
        sigma_stat_X = mc_err_elem[0]
        flag_abu[original_indices_X[0]] = 1
        if full_error_mode:
            param_errors = [array[mask] for array in param_errors]
            [sigma_T_X], [sigma_g_X], [sigma_v_X] = param_errors
            
    else: # nlin > 1
        clip_mask = np.ones(nlin, dtype=bool)
        
        if calc_flags == 0:
            if nlin >= 3:
                disp = mad_std(abu_elem)
                if np.isfinite(disp) and disp > 0.0:
                    clipped_abu = sigma_clip(abu_elem, sigma = sclip, cenfunc='median', stdfunc=mad_std)
                    if clipped_abu.count() > 0:
                        clip_mask = ~clipped_abu.mask
                        nlin = clipped_abu.count()
            
            final_indices_used = original_indices_X[clip_mask]
            flag_abu[final_indices_used] = 1 # Aggiorna il flag_abu
            
            final_indices_rejected = original_indices_X[~clip_mask]
            flag_abu[final_indices_rejected] = 0
                
        mean = np.mean(abu_elem[clip_mask]) - abu_sol - feh
        sigma_stat_X = np.std(abu_elem[clip_mask], ddof=1) / np.sqrt(nlin)
        
        if full_error_mode: #check se è consistente con il clipped_abu di sopra.
            abu_pT, abu_pg, abu_pv = (err[mask] for err in param_errors)
            sigma_T_X = np.mean(abu_pT[clip_mask])
            sigma_g_X = np.mean(abu_pg[clip_mask])
            sigma_v_X = np.mean(abu_pv[clip_mask])
    
    # 4. Controllo finale per valori NaN
    if np.isnan(mean):
        params_XH = {'abu_abs': 99.99, 's_tot': 99.99, 's_int': 99.99, 's_params': 99.99}   # FIX: was undefined (NameError)
        return 0, 99.99, 99.99, flag_abu, 99.99, 99.99, params_XH
        
    # 5. Calcola l'errore finale e restituisci in base alla modalità
    if full_error_mode:
        s_T_Fe, s_g_Fe, s_v_Fe = fe_errors
        variance = (s_stat_Fe**2 + sigma_stat_X**2 + (sigma_T_X - s_T_Fe)**2 + (sigma_g_X - s_g_Fe)**2 + (sigma_v_X - s_v_Fe)**2)
        final_sigma = np.sqrt(variance)
        sigma_params = np.sqrt((sigma_T_X - s_T_Fe)**2 + (sigma_g_X - s_g_Fe)**2 + (sigma_v_X - s_v_Fe)**2)
        sigma_int = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
        #error for [X/H]
        final_sigma_XH = np.sqrt(sigma_stat_X**2 + sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
        sigma_params_XH = np.sqrt(sigma_T_X**2 + sigma_g_X**2 + sigma_v_X**2)
        sigma_int_XH = sigma_stat_X
    else:
        if s_stat_Fe != 99.99:
            final_sigma = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
            sigma_params = 99.99
            sigma_int = np.sqrt(s_stat_Fe**2 + sigma_stat_X**2)
            #error for [X/H]
            final_sigma_XH = sigma_stat_X
            sigma_params_XH = 99.99
            sigma_int_XH = sigma_stat_X
        else:
            final_sigma = sigma_stat_X
            sigma_params = 99.99
            sigma_int = sigma_stat_X
            #error for [X/H]
            final_sigma_XH = sigma_stat_X
            sigma_params_XH = 99.99
            sigma_int_XH = sigma_stat_X

    mean_abs = np.floor((mean + abu_sol + feh) * 100 + 0.5) / 100
    params_XH = {'abu_abs': mean_abs, 's_tot': final_sigma_XH, 's_int': sigma_int_XH, 's_params': sigma_params_XH}
    return nlin, np.round(mean, 2), np.round(final_sigma, 2), flag_abu, np.round(sigma_params, 2), np.round(sigma_int, 2), params_XH


def update_flag_column(abu_file_path, flag_abu):
    # --- 1. read the original file ---
    try:
        with open(abu_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file {abu_file_path} not found.")
        return
        
    # --- 2. check of lengths ---
    num_data_rows = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    
    if len(flag_abu) != num_data_rows:
        print(f"Error: number of flags ({len(flag_abu)}) not corresponds to the number of line data ({num_data_rows})!")
        return
        
    # --- 3. process lines ---
    new_lines_content = []
    data_row_index = 0 # Contatore per scorrere l'array flag_abu
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            # Mantieni le righe vuote se ci sono
            new_lines_content.append('\n')
            continue
            
        if line_stripped.startswith('#'):
            # È un header o un commento: lo lasciamo intatto così com'è!
            # (Perché il nome della colonna flag c'è già nel file originale)
            new_lines_content.append(line)

        else:
            # È una riga di dati: la dividiamo nelle sue colonne
            parts = line_stripped.split()
            
            # Sostituiamo l'ultimo elemento (il vecchio flag) con il nuovo
            flag_value = int(flag_abu[data_row_index])
            parts[-1] = str(flag_value)
            
            # Ricostruiamo la riga unendo le colonne con una tabulazione
            new_data_line = "\t".join(parts) + "\n"
            new_lines_content.append(new_data_line)
            
            # Incrementa il contatore solo per le righe di dati
            data_row_index += 1
            
    # --- 4. Rewrite the original file ---
    try:
        with open(abu_file_path, 'w') as f:
            f.writelines(new_lines_content)
    except IOError as e:
        print(f"Error writing file: {e}")   
    

def merge_stellar_files(input_list_file):
    if not os.path.exists(input_list_file):
        print(f"Error: File {input_list_file} does not exist.")
        return

    with open(input_list_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    current_group = []
    for line in lines:
        if line == '###':
            process_group(current_group)
            current_group = []
        else:
            current_group.append(line)
            
    if current_group:
        process_group(current_group)


def process_group(group):
    if len(group) < 2:
        return

    # Splittiamo l'ultima riga per cercare l'eventuale [Fe/H]
    last_line_parts = group[-1].split()
    output_filename = last_line_parts[0]
    
    # Se la riga ha più di un elemento, il secondo è il nostro FeH esterno
    external_feh = None
    if len(last_line_parts) > 1:
        try:
            external_feh = float(last_line_parts[1])
        except ValueError:
            print(f"Attention: [Fe/H] not valid for {output_filename}")

    input_filenames = group[:-1]
    header_scritto = False

    try:
        with open(output_filename, 'w') as outfile:
            for fname in input_filenames:
                if os.path.exists(fname):
                    with open(fname, 'r') as infile:
                        for line in infile:
                            # Rimuoviamo gli spazi invisibili a inizio/fine per fare controlli sicuri
                            line_stripped = line.strip()
                            
                            # Saltiamo le righe completamente vuote per sicurezza
                            if not line_stripped:
                                continue
                            
                            # CONTROLLO HEADER: La riga inizia con '#' ?
                            if line_stripped.startswith('#'):
                                if not header_scritto:
                                    # Se è il primo file, copiamo l'header intatto
                                    outfile.write(line)
                                else:
                                    # Se abbiamo già scritto l'header, saltiamo tutte le righe con '#'
                                    continue
                            else:
                                # RIGA DI DATI: Se non inizia con '#', formattiamo e scriviamo i dati
                                columns = line.split()
                                new_line = "\t".join(columns) + "\n"
                                outfile.write(new_line)
                    
                    # Finito di elaborare tutto il primo file, l'header completo è stato scritto.
                    # Impostiamo il "semaforo" su True. Per i prossimi file gli header verranno ignorati!
                    header_scritto = True
                else:
                    print(f"Attention: File {fname} not found.")
        
        print(f"{output_filename} created.")
        if external_feh is not None:
            print(f"Using external [Fe/H]: {external_feh}")
        print('#'*40, '\n')

    except Exception as e:
        print(f"Error while elaborating {output_filename}: {e}")
        return
        
    if os.path.exists(output_filename):
        # Passiamo il valore esterno alla funzione di analisi
        analyze_abundances(output_filename, external_feh=external_feh)

def abu_absolute(elem, abu_fin, met):
    elem_tmp = elem.replace('II', '')
    abu_sol = abundance_link.get(elem_tmp)
    if elem_tmp != 'Fe':
        abu_abs = abu_fin + abu_sol + met
    else:   
        abu_abs = abu_fin + abu_sol
        
    return np.round(abu_abs, 2)


def get_header_names(file_path):
    """
    Legge un file e trova l'ultima riga di commento (#)
    per usarla come header.
    """
    header_line = None
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('#'):
                    # Continua a sovrascrivere, vogliamo l'ULTIMA riga di commento
                    header_line = line
                    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
        
    if header_line is None:
        print("Error: no header line (starting with #) found in the file.")
        return None
        
    # Pulisci la riga di header: rimuovi '#' e spazi extra, poi dividi
    column_names = header_line.lstrip('#').strip().split()
    return column_names

def get_coeffs(xx, yy):
    result = stats.linregress(xx, yy)
    
    m = result.slope
    c = result.intercept
    em = result.stderr
    ec = result.intercept_stderr 
        
    y_fit = m * xx + c
    return y_fit, m, c, em, ec

def apply_plot_style(ax, xlabel, ylabel):
    """
    Apply the requested aesthetic settings to a given axis (ax).
    """
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

    #  Impostazioni per i tick (indicatori)
    ax.tick_params(which='major', direction='in', length=8, top=True, right=True, labelsize=14)
    ax.tick_params(which='minor', direction='in', length=4, top=True, right=True)
    ax.minorticks_on()

    # Impostazioni per gli assi (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(1.75)


def making_plot(abu_file):
    # Plot Gala-like
    column_names = get_header_names(abu_file)
    if column_names:
        try:
            data = pd.read_csv(abu_file, sep=r"\s+", comment='#', header=None, names=column_names)
        except FileNotFoundError:
            print(f"Error: File not found '{abu_file}'")
            data = pd.DataFrame()
        
    else:
        print("No column names.")
        data = pd.DataFrame()   
    
    # --- 1. Data Selection ---
    fe_data = data[(data['ELEM'] == 26.00) & (data['ABU'] != 99.99)].copy()
    feii_data = data[(data['ELEM'] == 26.01) & (data['ABU'] != 99.99)].copy()

    # CONTROLLO PRINCIPALE: Se manca il FeI, non possiamo fare nulla.
    if not fe_data.empty:
        # --- Prepare Data for Plotting ---
        
        # 1. Calcoli per Fe I (Sappiamo che esiste)
        fe_data['reduced_ew'] = np.log10((fe_data['EW'].clip(lower=1e-9)/1000) / fe_data['LINE'])
        used_data = fe_data[fe_data['flag_line'] == 1]
        discarded_data = fe_data[fe_data['flag_line'] != 1]

        # 2. Calcoli per Fe II (SOLO SE ESISTE)
        has_feii = not feii_data.empty  # Variabile booleana di comodo
        
        if has_feii:
            feii_data['reduced_ew'] = np.log10((feii_data['EW'].clip(lower=1e-9)/1000) / feii_data['LINE'])
            used_feii = feii_data[feii_data['flag_line'] == 1]
            discarded_feii = feii_data[feii_data['flag_line'] != 1]
        
        # --- Calcolo Limiti Globali (FeI + FeII) ---
        
        # 1. Creiamo un set di dati unito solo per calcolare i min/max
        if not feii_data.empty:
            all_data_limits = pd.concat([fe_data, feii_data])
        else:
            all_data_limits = fe_data

        # 2. Calcoliamo i limiti usando questo set unito (all_data_limits)
        # Filtro per ABU e LINE (escludendo i valori 99.99)
        valid_rows_abu = all_data_limits[all_data_limits['ABU'] != 99.99]
        valid_abu = valid_rows_abu['ABU']
        valid_wv = valid_rows_abu['LINE']
        
        miny = valid_abu.min()
        maxy = valid_abu.max()
        minx1 = valid_wv.min()
        maxx1 = valid_wv.max()
        
        # Filtro per EP (escludendo i valori 99.999)
        valid_ep = all_data_limits[all_data_limits['EP'] != 99.999]['EP']
        minx2 = valid_ep.min()
        maxx2 = valid_ep.max()
        
        # --- Create the 3-Panel Plot ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6.75), sharey=True)
        
        # --- Panel 1: Wave ---
        # Fit solo su FeI
        if len(used_data['LINE']) >= 3:
            y_fit1, m1, c1, em1, ec1 = get_coeffs(used_data['LINE'], used_data['ABU'])
            ax1.plot(used_data['LINE'].to_numpy(), y_fit1.to_numpy(), c='red', label=rf'$y = ({m1:.3f} \pm {em1:.3f})x + ({c1:.2f} \pm {ec1:.2f})$')
            
        # Plot Fe I
        ax1.scatter(discarded_data['LINE'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, label='Discarded', lw=0.35)
        ax1.scatter(used_data['LINE'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', label='Used', lw=0.35)
        
        # Plot Fe II (SOLO SE ESISTE)
        if has_feii:
            ax1.scatter(discarded_feii['LINE'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
            ax1.scatter(used_feii['LINE'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', label='Fe II', lw=0.35)

        # Limiti e stile
        if np.isfinite(miny) and np.isfinite(maxy):
            ax1.set_ylim(miny - 0.2, maxy + 0.2)
            ax1.set_xlim(minx1 - 50, maxx1 + 50)
        else:
            ax1.set_ylim(0, 1)
        apply_plot_style(ax1, r'Wavelength [$\AA$]', 'A(Fe)')
        ax1.legend(fontsize=7)
        
        # --- Panel 2: EP ---
        # Fit FeI
        if len(used_data['LINE']) >= 3:
            y_fit2, m2, c2, em2, ec2 = get_coeffs(used_data['EP'], used_data['ABU'])
            ax2.plot(used_data['EP'].to_numpy(), y_fit2.to_numpy(), c='red', label=rf'$y = ({m2:.2f} \pm {em2:.2f})x + ({c2:.2f} \pm {ec2:.2f})$')

        # Plot Fe I
        ax2.scatter(discarded_data['EP'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, lw=0.35)
        ax2.scatter(used_data['EP'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', lw=0.35)
        
        # Plot Fe II (SOLO SE ESISTE)
        if has_feii:
            ax2.scatter(discarded_feii['EP'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
            ax2.scatter(used_feii['EP'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', lw=0.35)

        ax2.set_xlim(minx2 - 0.2, maxx2 + 0.2)
        apply_plot_style(ax2, 'Excitation Potential [eV]', '')
        ax2.legend(fontsize=7)
        
        # --- Panel 3: Reduced EW ---
        # Fit FeI
        if len(used_data['LINE']) >= 3:
            y_fit3, m3, c3, em3, ec3 = get_coeffs(used_data['reduced_ew'], used_data['ABU'])
            ax3.plot(used_data['reduced_ew'].to_numpy(), y_fit3.to_numpy(), c='red', label=rf'$y = ({m3:.2f} \pm {em3:.2f})x + ({c3:.2f} \pm {ec3:.2f})$')

        # Plot Fe I
        ax3.scatter(discarded_data['reduced_ew'].to_numpy(), discarded_data['ABU'].to_numpy(), color='gray', alpha=0.5, lw=0.35)
        ax3.scatter(used_data['reduced_ew'].to_numpy(), used_data['ABU'].to_numpy(), color='blue', ec='k', lw=0.35)
        
        ew_series_list = [used_data['reduced_ew'], discarded_data['reduced_ew']]
        
        # Plot Fe II (SOLO SE ESISTE)
        if has_feii:
            ax3.scatter(discarded_feii['reduced_ew'].to_numpy(), discarded_feii['ABU'].to_numpy(), color='lightcoral', marker='^', alpha=0.5, lw=0.35)
            ax3.scatter(used_feii['reduced_ew'].to_numpy(), used_feii['ABU'].to_numpy(), color='red', ec='k', marker='^', lw=0.35)
            
            ew_series_list.extend([used_feii['reduced_ew'], discarded_feii['reduced_ew']])
            
        # --- CALCOLO MIN E MAX ---
        all_ew = pd.concat(ew_series_list)
        
        if not all_ew.empty:
            ewr_min = all_ew.min()
            ewr_max = all_ew.max()
        else:
            ewr_min, ewr_max = -6.0, -4.4
        
        ax3.set_xlim(ewr_min - 0.1, ewr_max + 0.1)
        apply_plot_style(ax3, r'log(EW / $\lambda$)', '')
        ax3.legend(fontsize=7)
        
        # --- Finalization ---
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.75, wspace=0.1)
        
        name_star = abu_file.split('.')[0]
        plt.savefig(name_star+'.pdf')   
        plt.close() 


        

def analyze_abundances(abu_file, external_feh=None):
    try:
        id_from_filename = os.path.basename(abu_file).replace('.fit', '')
    
        col_names_base = ['ID', 'line', 'nelem', 'abu', 'ew', 'EP', 'loggf', 'abu_mc']
        col_names_err = ['abu_pT', 'abu_pg', 'abu_pv']
        
        num_cols = 0 
        with open(abu_file, "r") as f:
            for line in f:
                # Puliamo la riga da eventuali spazi invisibili a inizio o fine
                line_stripped = line.strip()
        
                # Controlliamo se è la riga che ci interessa
                if line_stripped.startswith("#SPECTRUM"):
                    # Rimuoviamo il simbolo '#' per avere solo il testo delle colonne
                    header_pulito = line_stripped.replace('#', '')
            
                    # Tagliamo la riga in parole singole e le contiamo
                    num_cols = len(header_pulito.split())
            
                   # Usciamo dal ciclo: abbiamo trovato le colonne, non serve leggere i dati sotto!
                    break      
                          
        if num_cols == 12:
            # 8 basi + 3 errori + 1 flag in fondo
            all_col_names = col_names_base + col_names_err + ['flag']
        elif num_cols == 9:
            # 8 basi + 1 flag in fondo
            all_col_names = col_names_base + ['flag']
        else:
            # Caso di fallback (es. se file malformato)
            all_col_names = col_names_base[:num_cols]
        
        df = pd.read_csv(abu_file, sep=r"\s+", comment='#', header=None, names=all_col_names)

        if df.empty: return

        ID = id_from_filename
        nelem = df['nelem'].to_numpy(copy=True)
        abu = df['abu'].to_numpy(copy=True)
        abu_mc = df['abu_mc'].to_numpy(copy=True)         
        flag_line = df['flag'].to_numpy(copy=True)
        lines_to_write = []
        unique_nelems = np.unique(nelem)
        other_nelems = unique_nelems[unique_nelems != 26.00]
        
        ### calcolo flag_line in base ad EWR ed EP
        EW = df['ew'].to_numpy(copy=True)
        EP = df['EP'].to_numpy(copy=True)
        waves = df['line'].to_numpy(copy=True)  
        EWR = np.log10((np.maximum(EW, 1e-9) / 1000) / waves)
        
        if calc_flags == 0:
            # FIX: start again from all the lines with an abundance (before, the -1/-2 flags of the old
            # limits were kept, so those lines stayed rejected with the new limits)
            flag_line = np.where(abu != 99.99, 0, flag_line)
            # condizione su EWR
            condizione_abu = (abu != 99.99) 
            condizione_ewr = (EWR < ewr_min) | (EWR > ewr_max)
        
            mask_ewr = condizione_abu & condizione_ewr
            flag_line[mask_ewr] = -1
        
            # condizione su EP
            condizione_ep_range = (EP < ep_min) | (EP > ep_max)
            condizione_ep_valido = (EP != 99.999)
        
            mask_ep = condizione_abu & condizione_ep_range & condizione_ep_valido
            flag_line[mask_ep] = -2
                        
        # --- GESTIONE FERRO (26.00) ---
        # Calcoliamo comunque il ferro se presente per avere i flag_line corretti
        has_fe = 26.00 in nelem
        feh_calc = 99.99
        s1_fe = 0.0
        
        # Prepariamo gli errori se presenti
        p_errs = (df['abu_pT'].to_numpy(copy=True),
                  df['abu_pg'].to_numpy(copy=True),
                  df['abu_pv'].to_numpy(copy=True)) if 'abu_pT' in df.columns else None

        if has_fe:
            if p_errs:
                nlin_fe, feh_calc, sigma_fe, s1_fe, s2, s3, s4, flag_line, s_param = extract_abu_fe(nelem, abu, abu_mc, 26.00, flag_line, *p_errs)
                fe_err_tuple = (s2, s3, s4)
            else:
                nlin_fe, feh_calc, s1_fe, flag_line, s_param = extract_abu_fe(nelem, abu, abu_mc, 26.00, flag_line)
                sigma_fe = s1_fe
                fe_err_tuple = None
                
        else:
            nlin_fe, sigma_fe, s_param = 0, 99.99, 99.99
            fe_err_tuple = (0.0, 0.0, 0.0) if p_errs else None

        # LOGICA PRIORITÀ: Se external_feh è fornito, sovrascrive quello calcolato
        if external_feh is not None:
            feh_final = external_feh
            nlin_fe = 0
            sigma_fe = 99.99
            s1_fe = 99.99
            s_param = 99.99
        else:
            feh_final = feh_calc
        
        #feh_final = external_feh if external_feh is not None else feh_calc
        abu_abs_fe = abu_absolute('Fe', feh_final, feh_final)

        # Scriviamo la riga del Ferro (usando il valore finale)
        lines_to_write.append(f"{ID:<30}  {26.00:5.2f}  {nlin_fe:>5}  {feh_final:>5.2f}  {sigma_fe:>5.2f}  {s1_fe:>5.2f}  {s_param:>5.2f}       {abu_abs_fe:>5.2f}  {sigma_fe:>5.2f}   {s1_fe:>5.2f}   {s_param:>5.2f}")

        # --- GESTIONE ALTRI ELEMENTI ---
        for nel in other_nelems:
            if nel == 26.01:
                if p_errs:
                    nlin, mean, sigma, s_int, *_, flag_line, s_p = extract_abu_fe(nelem, abu, abu_mc, nel, flag_line, *p_errs)
                else:
                    nlin, mean, sigma, flag_line, s_p = extract_abu_fe(nelem, abu, abu_mc, nel, flag_line)
                    s_int = sigma
                   
                abu_abs_xh = abu_absolute('Fe', mean, feh_final)
                s_tot_xh = sigma
                s_int_xh = s_int
                s_params_xh = s_p
            else:
                # Usiamo feh_final per scalare [X/Fe]
                nlin, mean, sigma, flag_line, s_p, s_int, params_XH = extract_abu(nelem, abu, abu_mc, nel, feh_final, flag_line, s1_fe, p_errs, fe_err_tuple)
                
                abu_abs_xh = params_XH['abu_abs']
                s_tot_xh = params_XH['s_tot']
                s_int_xh = params_XH['s_int']
                s_params_xh = params_XH['s_params'] 

            lines_to_write.append(f"{ID:<30}  {nel:>5.2f}  {nlin:>5}  {mean:>5.2f}  {sigma:>5.2f}  {s_int:>5.2f}  {s_p:>5.2f}       {abu_abs_xh:>5.2f}  {s_tot_xh:>5.2f}   {s_int_xh:>5.2f}   {s_params_xh:>5.2f}")

        # Scrittura e flag (come prima)
        with open('abundance_final_merged.txt', 'a+') as f:
            for l in lines_to_write: f.write(l + '\n')
            f.write('#' + '-'*39 + '\n')
        
        update_flag_column(abu_file, flag_line)
        
        # makes the 3 panel plot
        making_plot(abu_file)

    except Exception as e:
        print(f"Error during the analysis of {abu_file}: {e}")


def get_type_input(prompt, target_type):
    while True:
        try:
            value = input(prompt)
            return target_type(value)
        except ValueError:
            # Se fallisce, avvisa l'utente e il ciclo riparte in automatico
            print(f"Error: please enter a valid value (required type: {target_type.__name__}).")


def main() -> int:
    """Entry point of salvador-merge: asks for the list file and the settings."""
    global calc_flags, sclip, ep_min, ep_max, ewr_min, ewr_max, abundance_link, symbol_map
    to_merge = get_type_input('Filename containing the .fit files to merge: ', str)
    print('')
    solar_ref = get_type_input('Which solar reference? (1 -> CL | 2 -> GS98 | 3 -> Magg22 | 4 -> Asp09 ):  ', int)
    print('')
    calc_flags = get_type_input('Recalculate the flags (0) or use the flags present in the .fit files (1)?:  ', int)
    print('')
    
    solar_ref = int(solar_ref + 1)
    
    if calc_flags == 0:
        print("\n--- Parameters for sigma, EWR, and EP clipping ---")
        sclip = get_type_input('Insert value for sigma clipping? ', float)
        ep_min = get_type_input("Insert EP min (es. 0.0): ", float)
        ep_max = get_type_input("Insert EP max (es. 10.0): ", float)
        ewr_min = get_type_input("Insert EWR min (es. -6.0): ", float)
        ewr_max = get_type_input("Insert EWR max (es. -4.0): ", float)
        print('')

    solar_data = read_solar_data('solar_abu.txt')
    if solar_data:
        abundance_link = abu_dict(solar_data, column=solar_ref)
        symbol_map = {int(line[0]): line[1] for line in solar_data}
    else:
        print('Mandatory file solar_abu.txt is missing!!!')
        sys.exit()

    # Inizializza il file dei risultati finali (pulisce se esiste già)
    with open('abundance_final_merged.txt','w') as f:
        f.write('#STAR\t\t= name of the spectrum \n')
        f.write('#CODE\t\t= element code in kurucz format \n')
        f.write('#N_LIN\t\t= number of lines used to compute the mean \n')
        f.write('#ABU\t\t= abundance in the format [Fe/H] or [X/Fe] \n')
        f.write('#ERROR\t\t= total error of ABU. Sum in quadrature of of E_INT + E_PARAM \n')
        f.write('#E_INT\t\t= internal error. Sigma / sqrt(N_LIN) or MC error \n')
        f.write('#E_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n')
        f.write('#A_ABS\t\t= absolute abundance \n')
        f.write('#E_X/H\t\t= total error of abundance in [X/H] format. Sum in quadrature of of E_X/H_INT + E_X/H_PARAM \n')
        f.write('#E_X/H_INT\t= internal error. Sigma / sqrt(N_LIN) or MC error \n')
        f.write('#E_X/H_PARAM\t= atmospheric params error. Sum in quadrature of error due to Teff, logg, and vt \n')
        f.write('#' + '-'*110 + '\n')
        f.write('#STAR                            CODE  N_LIN    ABU  ERROR  E_INT  E_PARAM     A_ABS   E_XH E_XH_INT E_XH_PARAM\n')


    merge_stellar_files(to_merge)
    return 0


if __name__ == "__main__":
    sys.exit(main())

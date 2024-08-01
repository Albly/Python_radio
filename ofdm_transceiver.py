import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# TODO: Exclude following libs in next release:
import sionna as sn
import tensorflow as tf

def create_preamble(N_FFT, CP_len, N_repeat = 2):
    '''
    Creates Preamble for synchronization and CFO compensation
    Uses Schmidl's OFDM preamble design scheme

    TODO: from https://liusc1028.github.io/papers/ISPLC13.pdf
    implement Minn's and Proposed methods
    '''
    # random BPSK symbols
    preamble = 1 - 2 * np.random.randint(0, 2, size=(int(N_FFT/2), 1))
    preamble = np.complex64(preamble)
    # repeat preamble
    preamble_full = np.tile(preamble, (N_repeat, 1))
    # add cyclic prefix
    preamble_full_cp = np.concatenate((preamble_full[-CP_len:], preamble_full))
    return preamble_full_cp

def generate_bits(N_subcarriers: int, N_bits_per_conts_point:int):
    """generates random sequence of bits with defined size

    Args:
        N_subcarriers (int): Number of subcaarriers used for data transmission
        N_bits_per_conts_point (int): Number of bits in QAM symbol

    Returns:
        ndarray: returns bits 
    """
    data_stream =  np.random.binomial(n = 1, p = 0.5, size = (N_subcarriers*N_bits_per_conts_point))
    return data_stream

def qam_modulate(bits, num_bits_per_symbol):
    """Applies QAM modulation for provided bitsteam
    TODO Temporary algorithm! Reimplement same modem with numpy logic.

    Args:
        bits (ndarray): array of bits with shape of [1 x N_symbols*num_bits_per_symbol]
        num_bits_per_symbol (int): number of bits per QAM symbol

    Returns:
        ndarray: Normalized QAM symbols with shape of [1 x N_symbols]
    """
    constellation = sn.mapping.Constellation("qam", num_bits_per_symbol)
    mapper = sn.mapping.Mapper(constellation = constellation)
    return mapper(tf.convert_to_tensor(bits[None,...])).numpy()

def qam_demodulate(signal, num_bits_per_symbol):
    """ QAM demodulator with soft decision. 
    TODO Temporary algorithm! Reimplement same modem with numpy logic.

    Args:
        signal (ndarray): aatay with QAM symbols with shape of [1 x N_symbols]
        num_bits_per_symbol (int): number of bits per QAM symbol

    Returns:
        ndarray: demodulated bitstream with shape of [1 x N_symbols*num_bits_per_symbol]
    """
    constellation = sn.mapping.Constellation("qam", num_bits_per_symbol)
    demapper = sn.mapping.Demapper("app", constellation = constellation)
    return (demapper([tf.convert_to_tensor(signal), 0.0]).numpy() > 0).astype(np.int64)

def form_baseband_ofdm_symb(qam_symb, N_FFT, CP_len, dc_offset = False):
    """Form timedomain baseband OFDM symbol using input QAM symbols 

    Args:
        qam_symb (ndarray): array of QAM symbols
        N_FFT (int): Number of FFT samples
        CP_len (int): Cyclic prefix length
        dc_offset (bool, optional): Whether use DC component for allocating data. Defaults to False.

    Returns:
        ndarray: time-domain baseband OFDM symbol with cyclic prefix
    """
    qam_symb = qam_symb.ravel()
    N_subcarriers = qam_symb.shape[0]
    assert N_subcarriers < N_FFT, 'Cannot send more symbols than FFT size'
    dc = int(dc_offset)
    
    # Frequency-domain baseband OFDM symbol
    tx_ofdm_symb = np.zeros((N_FFT), dtype = np.complex64)
    tx_ofdm_symb[dc : N_subcarriers//2 + dc] = qam_symb[N_subcarriers//2:]
    tx_ofdm_symb[-N_subcarriers//2:] = qam_symb[:N_subcarriers//2]

    # Time-domain baseband OFDM symbol 
    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_symb, axis = 0, norm = 'ortho')
    time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )
    return time_ofdm_sym_cp_pilot


def create_data(N_sc, N_fft, CP_len, dc_offset = False, aditional_return = None):
    
    data_stream =  np.random.randint(0, 2, size=(N_sc, 1))
    data_stream_mod = 1 - 2 * data_stream
    mod_sym_pilot = np.complex64(data_stream_mod)

    tx_ofdm_sym = np.zeros((N_fft, 1), dtype = np.complex64)
    dc = int(dc_offset)
    tx_ofdm_sym[dc : N_sc//2 + dc] = mod_sym_pilot[ N_sc//2: ]
    tx_ofdm_sym[-N_sc//2: ] = mod_sym_pilot[ 0 : N_sc//2]

    time_ofdm_sym_pilot = np.fft.ifft(tx_ofdm_sym, axis = 0, norm = 'ortho')
    time_ofdm_sym_cp_pilot = np.concatenate( (time_ofdm_sym_pilot[-CP_len:], time_ofdm_sym_pilot) )

    if aditional_return == 'mod_sym_pilot':
        return time_ofdm_sym_cp_pilot, mod_sym_pilot
    if aditional_return == 'data_stream':
        return time_ofdm_sym_cp_pilot, data_stream

#############################################################################
def find_edges(rx_sig, frame_len, preamble_len, start_idx):
    corr_list = []
    for idx_search in range(start_idx, start_idx + frame_len):
        
        first_part = rx_sig[idx_search : idx_search + preamble_len]
        second_part = rx_sig[idx_search + preamble_len : idx_search + 2*preamble_len]
        
        corr_now = np.dot(np.conj(first_part), second_part)        
        corr_list.append(corr_now)

    rel_idx = np.argmax(np.abs(corr_list), axis = 0)
    idx_max = rel_idx + start_idx

    return idx_max, corr_list[rel_idx]

def cfo(frame_receive, corr_value, preamble_len):
    angle_cfo = np.angle(corr_value) / preamble_len
    cfo_comp_sig = np.exp(1j * (-angle_cfo * np.arange(0, frame_len)) )
    frame_receive = frame_receive * cfo_comp_sig
    return frame_receive

def baseband_freq_domian(pilot_freq, N_sc_use):
    rec_sym_pilot = np.zeros((N_sc_use,1), dtype = np.complex64)
    rec_sym_pilot[int(N_sc_use/2):, 0] = pilot_freq[0+dc_offset:int(N_sc_use/2) + dc_offset]
    rec_sym_pilot[0:int(N_sc_use/2), 0] = pilot_freq[-int(N_sc_use/2):]
    
    return rec_sym_pilot

def channel_estimation(h_ls, CP_len, N_fft):
    h_time = np.fft.ifft(h_ls, N_fft, 0, norm='ortho')
    ce_len = len(h_ls)

    W_spead = int(CP_len/8)
    W_sync_err = int(CP_len)
    W_max = W_spead + W_sync_err
    W_min = W_sync_err

    eta_denoise = np.zeros_like(h_time)
    eta_denoise[-W_min:] = 1.0 
    eta_denoise[0:W_max] = 1.0

    h_time_denoise = h_time * eta_denoise

    h_hw = np.fft.fft(h_time_denoise, N_fft, 0, norm='ortho')
    h_ls = h_hw[0:ce_len]


    display.clear_output(wait = True)
    plt.figure(100)
    plt.plot(np.arange(0, N_fft), np.abs(h_time))
    plt.plot(np.arange(0, N_fft), np.abs(h_time_denoise))
    plt.title('Channel response time domain')
    plt.grid(); plt.show()

    return h_ls

def estimate_SNR(pilot_freq,rec_sym_pilot,N_sc_use):
    noise_arr = pilot_freq[int(N_sc_use/2) : -int(N_sc_use/2)]
    sigma0_freq = np.real( np.dot( np.conj(noise_arr) , noise_arr ) ) / len(noise_arr)
    Es_freq = np.real( np.dot( np.conj(rec_sym_pilot[:,0]) , rec_sym_pilot[:,0] ) ) / len(rec_sym_pilot[:, 0])
    SNR_est = 10.0 * np.log10(Es_freq / sigma0_freq)
    return SNR_est

def demodulate(eq_data, N_sc_use):
    bit_arr = []    
    for idx in range(0, N_sc_use):
        
        if (np.real(eq_data[idx]) > 0):
            bit_curr = 0
        else:
            bit_curr = 1
            
        bit_arr.append(bit_curr)
    return bit_arr

def get_ber(data_stream, bit_arr, N_sc_use):
    err_num = 0
    for idx in range(0, N_sc_use):
        if (data_stream[idx, 0] != bit_arr[idx]):
            err_num = err_num + 1
    ber = 1.0 * err_num / N_sc_use
    return ber

#############################################################################

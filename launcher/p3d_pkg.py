#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Extract PS4 .pkg images (the CUSA12636 game) into the Image0/ + Sc0/
layout the PKG extraction tools use: the PFS files (eboot.bin, right.sprx, data/,
sce_module/, ...) go to dest/Image0/ and the PKG entry files (param.sfo, icon0.png,
pic1.png, trophy/, ...) go to dest/Sc0/. The launcher adopts an Sc0/ beside an Image0/
into a game folder's sce_sys (p3d_launcher_win.system_files_beside), so an extracted
.pkg is picked up by the launcher exactly like a folder made with a PKG tool.

No third-party modules are required: the FSELF/CNT container, the PKG entry table,
the PFS3/PFSC image and the XTS-AES-128 payload are pure Python (standard library).
If the optional `cryptography` package is installed, XTS runs at C speed (a 5 GiB
game in under a minute); without it the pure-Python path is used (about 0.5 MiB/s:
a full game takes hours, and the progress report says which path is running).

Format references (format only; no code is taken from them):
  * the PS4 PKG / FSELF / CNT container, the PFS3 superblock and the PFSC layout,
    and the PKG entry names, as documented in the ShadPKG project (seregonwar/ShadPKG,
    core/file_format/pkg.h, pfs.h, pkg_type.cpp; GPL-2.0-or-later) and the psdevwiki
    PKG and PFS pages.
  * XTS-AES-128 in the variant the PS4 PFS images use (the mbedTLS scheme: the
    encrypted tweak S_j = AES_K2(i) is advanced by x in GF(2^128) per 16-byte
    block, a 4096-byte sector carries 256 blocks, and the 128-bit initial tweak
    is the little-endian sector index). When the pure-Python path is used it is
    checked against the FIPS-197 AES-128 vector and the IEEE P1619/D16 XTS-A128
    test vectors before first use.
  * the two RSA keys are the public `pkg_entry_key_3` and `fake_ekpfs_key` constants
    shipped with the community PKG tools (flatz/pkg_pfs_tool, ShadPKG's config.ini).
  * the NPDRM PKGs: entries 0x400..0x403 are AES-128-CBC ciphertexts keyed per entry
    (ShadPKG's ivKey scheme, documented in its docs/HOWWORKS.md), and the entry bytes
    are zero-padded to a 16-byte boundary before decryption, then trimmed back to the
    entry size (npbind.dat is 532 bytes, not a multiple of 16).

All offsets are checked before use.
"""
import argparse
import hashlib
import hmac
import io
import mmap
import os
import pathlib
import shutil
import struct
import sys
import zlib

BLOCK = 0x10000           # PFS/PFSC logical block
XTS_SECTOR = 4096         # XTS sector: 256 blocks
MAGIC_CNT = b'\x7fCNT'
MAGIC_PFS = 0x1332A0B    # PFS3 superblock magic (little-endian u64 at superblock + 8)
MAGIC_PFSC = 0x43534650  # PFSCHdr magic, 'PFSC' (little-endian u32)
MODE_SIGNED = 0x1
MODE_ENCRYPTED = 0x4
INODE_STRIDE = 0xA8      # the inodes in the PFSC inode-table blocks

# The PKG entry types the launcher's game layout uses, and where each lands in Sc0/
# (the names follow ShadPKG's pkg_type.cpp; the trophy files are numbered 00..99).
SC0_NAMES = {
    0x1000: 'param.sfo',
    0x1001: 'playgo-chunk.dat',
    0x1002: 'playgo-chunk.sha',
    0x1003: 'playgo-manifest.xml',
    0x1004: 'pronunciation.xml',
    0x1005: 'pronunciation.sig',
    0x1006: 'pic1.png',
    0x100B: 'shareparam.json',
    0x1200: 'icon0.png',
    0x1220: 'pic0.png',
    0x1240: 'livearea/snd0.wav',
    0x1280: 'livearea/icon0.dds',
    0x12A0: 'livearea/pic0.dds',
    0x12C0: 'livearea/pic1.dds',
}
NPDRM_NAMES = {0x400: 'npdrm/license.dat', 0x401: 'npdrm/license.info',
               0x402: 'npdrm/nptitle.dat', 0x403: 'npdrm/npbind.dat'}


def sc0_name(entry_type):
    """The Sc0/ relative name of a PKG entry, or None when it is not a game file."""
    if entry_type in NPDRM_NAMES:
        return NPDRM_NAMES[entry_type]
    if entry_type in SC0_NAMES:
        return SC0_NAMES[entry_type]
    if 0x1400 <= entry_type <= 0x1463:
        return f'trophy/trophy{entry_type - 0x1400:02d}.trp'
    return None


class PkgError(Exception):
    """A .pkg the module cannot handle (bad container, missing keys, unknown layout)."""


# --------------------------------------------------------------------------- AES-128
# A pure-Python AES-128 and the XTS-AES-128 the PFS images use. The S-box is generated
# from its FIPS-197 definition (GF(2^8) inverse + affine map) so it is correct by
# construction; the block cipher and the XTS are checked against the FIPS-197 and
# IEEE P1619/D16 test vectors before first use (see _selftest).

def _xtime(a):
    a <<= 1
    if a & 0x100:
        a ^= 0x11B
    return a & 0xFF


def _gf_mul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


def _gf_inv(a):
    if not a:
        return 0
    r, e, x, y = 1, 254, a, 1
    while e:
        if e & 1:
            r = _gf_mul(r, x)
        x = _gf_mul(x, x)
        e >>= 1
    return r


def _sbox_affine(b):
    """FIPS-197 affine map on the 8 input bits: y_i = b_i ^ b_{i+4} ^ b_{i+5} ^
    b_{i+6} ^ b_{i+7} ^ c_i, c = 01100011b (c_i is its i-th bit)."""
    out = 0
    for i in range(8):
        bit = (b >> i) & 1
        bit ^= (b >> ((i + 4) & 7)) & 1
        bit ^= (b >> ((i + 5) & 7)) & 1
        bit ^= (b >> ((i + 6) & 7)) & 1
        bit ^= (b >> ((i + 7) & 7)) & 1
        bit ^= (0x63 >> i) & 1
        out |= bit << i
    return out


def _build_sbox():
    return [_sbox_affine(_gf_inv(x)) for x in range(256)]


_SBOX = _build_sbox()
_ISBOX = [0] * 256
for _i, _v in enumerate(_SBOX):
    _ISBOX[_v] = _i


def _gmul_table(c):
    return [_gf_mul(i, c) for i in range(256)]


_XT = _gmul_table(0x02)
_G9, _GB, _GD, _GE = (_gmul_table(0x09), _gmul_table(0x0B),
                      _gmul_table(0x0D), _gmul_table(0x0E))


def _expand_key_128(key):
    """A 16-byte key -> 11 round keys, each a 16-byte (column-major) layout."""
    n_k, n_r = 4, 10
    rcon = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)
    w = [list(key[4 * i:4 * i + 4]) for i in range(n_k)]
    for i in range(n_k, n_k * (n_r + 1)):
        t = list(w[i - 1])
        if i % n_k == 0:
            t = t[1:] + t[:1]
            t = [_SBOX[b] for b in t]
            t[0] ^= rcon[i // n_k - 1]
        w.append([w[i - n_k][j] ^ t[j] for j in range(4)])
    rounds = []
    for r in range(n_r + 1):
        rk = bytearray(16)
        for c in range(4):
            rk[4 * c:4 * c + 4] = w[4 * r + c]
        rounds.append(bytes(rk))
    return rounds


def _to_state(block):
    return [[block[4 * c + r] for c in range(4)] for r in range(4)]


def _from_state(st):
    out = bytearray(16)
    for c in range(4):
        for r in range(4):
            out[4 * c + r] = st[r][c]
    return bytes(out)


def _aes_enc(block, rounds):
    st = _to_state(block)
    for r in range(4):
        for c in range(4):
            st[r][c] ^= rounds[0][r + 4 * c]
    for rnd in range(1, 10):
        for r in range(4):
            for c in range(4):
                st[r][c] = _SBOX[st[r][c]]
        st[1] = st[1][1:] + st[1][:1]
        st[2] = st[2][2:] + st[2][:2]
        st[3] = st[3][3:] + st[3][:3]
        for c in range(4):
            a0, a1, a2, a3 = st[0][c], st[1][c], st[2][c], st[3][c]
            st[0][c] = _XT[a0] ^ (a1 ^ _XT[a1]) ^ a2 ^ a3
            st[1][c] = a0 ^ _XT[a1] ^ (a2 ^ _XT[a2]) ^ a3
            st[2][c] = a0 ^ a1 ^ _XT[a2] ^ (a3 ^ _XT[a3])
            st[3][c] = (a0 ^ _XT[a0]) ^ a1 ^ a2 ^ _XT[a3]
        for r in range(4):
            for c in range(4):
                st[r][c] ^= rounds[rnd][r + 4 * c]
    for r in range(4):
        for c in range(4):
            st[r][c] = _SBOX[st[r][c]]
    st[1] = st[1][1:] + st[1][:1]
    st[2] = st[2][2:] + st[2][:2]
    st[3] = st[3][3:] + st[3][:3]
    for r in range(4):
        for c in range(4):
            st[r][c] ^= rounds[10][r + 4 * c]
    return _from_state(st)


def _aes_dec(block, rounds):
    """AES-128 block decrypt (equivalent inverse cipher) with the forward round keys."""
    st = _to_state(block)
    for r in range(4):
        for c in range(4):
            st[r][c] ^= rounds[10][r + 4 * c]
    for rnd in range(9, 0, -1):
        st[1] = st[1][3:] + st[1][:3]
        st[2] = st[2][2:] + st[2][:2]
        st[3] = st[3][1:] + st[3][:1]
        for r in range(4):
            for c in range(4):
                st[r][c] = _ISBOX[st[r][c]]
        for r in range(4):
            for c in range(4):
                st[r][c] ^= rounds[rnd][r + 4 * c]
        for c in range(4):
            a0, a1, a2, a3 = st[0][c], st[1][c], st[2][c], st[3][c]
            st[0][c] = _GE[a0] ^ _GB[a1] ^ _GD[a2] ^ _G9[a3]
            st[1][c] = _G9[a0] ^ _GE[a1] ^ _GB[a2] ^ _GD[a3]
            st[2][c] = _GD[a0] ^ _G9[a1] ^ _GE[a2] ^ _GB[a3]
            st[3][c] = _GB[a0] ^ _GD[a1] ^ _G9[a2] ^ _GE[a3]
    st[1] = st[1][3:] + st[1][:3]
    st[2] = st[2][2:] + st[2][:2]
    st[3] = st[3][1:] + st[3][:1]
    for r in range(4):
        for c in range(4):
            st[r][c] = _ISBOX[st[r][c]]
    for r in range(4):
        for c in range(4):
            st[r][c] ^= rounds[0][r + 4 * c]
    return _from_state(st)


def _aes_cbc_decrypt(data, key, iv):
    """AES-128-CBC block decrypt (pure Python; only the small PKG entries use it)."""
    rounds = _expand_key_128(key)
    out = bytearray()
    prev = iv
    for i in range(0, len(data), 16):
        blk = data[i:i + 16]
        out += bytes(a ^ b for a, b in zip(_aes_dec(blk, rounds), prev))
        prev = blk
    return bytes(out)


def _mul_x(t):
    """Multiply the little-endian 128-bit field element `t` by x in GF(2^128)
    (the 0x87 polynomial; mbedtls_gf128mul_x_ble): the low half's bit 63 carries
    into the high half, and the high half's bit 63 reduces to 0x87 in the low half."""
    m64 = (1 << 64) - 1
    lo = int.from_bytes(t[:8], 'little')
    hi = int.from_bytes(t[8:], 'little')
    carry = lo >> 63
    lo = ((lo << 1) & m64) ^ (0x87 if (hi >> 63) & 1 else 0)
    hi = ((hi << 1) | carry) & m64
    return lo.to_bytes(8, 'little') + hi.to_bytes(8, 'little')


def _xts_blocks(blocks, rk1, rk2, sector_index, decrypt=True):
    """XTS-AES-128 over one sector's 16-byte blocks (the last of which may be shorter
    and is zero-padded): D_K1(C_j ^ S_j) ^ S_j to decrypt (A_K1 when decrypt=False),
    S_0 = AES_K2(i), S_j+1 = xS_j."""
    s = _aes_enc(sector_index.to_bytes(16, 'little'), rk2)
    out = bytearray()
    for blk in blocks:
        blk = blk.ljust(16, b'\0')
        xored = bytes(a ^ b for a, b in zip(blk, s))
        payload = _aes_dec(xored, rk1) if decrypt else _aes_enc(xored, rk1)
        out += bytes(a ^ b for a, b in zip(payload, s))
        s = _mul_x(s)
    return bytes(out)


def _xts_sector_pure(data, rk1, rk2, sector_index):
    """One 4096-byte XTS sector (or its available tail), pure Python."""
    return _xts_blocks([data[j * 16:(j + 1) * 16] for j in range(XTS_SECTOR // 16)],
                       rk1, rk2, sector_index)


# The optional `cryptography` package gives C-speed AES-XTS: its XTS mode takes the
# combined 32-byte key (data key || tweak key) and the 128-bit initial tweak.
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    Cipher = None


def _xts_sector_fast(data, key32, sector_index):
    dec = Cipher(algorithms.AES(key32),
                 modes.XTS(sector_index.to_bytes(16, 'little'))).decryptor()
    return dec.update(data) + dec.finalize()


def _selftest():
    """The pure-Python AES/XTS, checked once before first use: the FIPS-197 AES-128
    vector, the three IEEE P1619/D16 XTS-A128 Annex B vectors (the ones mbedTLS's own
    self-test uses: 32-byte sectors) in both directions, and a 4096-byte round-trip."""
    key = bytes(range(16))
    pt = bytes.fromhex('00112233445566778899aabbccddeeff')
    ct = bytes.fromhex('69c4e0d86a7b0430d8cdb78070b4c55a')
    rk = _expand_key_128(key)
    if _aes_enc(pt, rk) != ct or _aes_dec(ct, rk) != pt:
        raise PkgError('AES-128 self-test failed (FIPS-197 vector)')
    tweak = int.from_bytes(bytes.fromhex('33333333330000000000000000000000'), 'little')
    vecs = [
        (bytes(32), bytes(16), bytes(16), 0,
         '917cf69ebd68b2ec9b9fe9a3eadda692cd43d2f59598ed858c02c2652fbf922e'),
        (bytes([0x44] * 32), bytes([0x11] * 16), bytes([0x22] * 16),
         tweak, 'c454185e6a16936e39334038acef838bfb186fff7480adc4289382ecd6d394f0'),
        (bytes([0x44] * 32),
         bytes.fromhex('fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0'),
         bytes([0x22] * 16),
         tweak, 'af85336b597afc1a900b2eb21ec949d292df4c047e0b21532186a5971a227a89'),
    ]
    for pt, k1, k2, idx, ref in vecs:
        blocks = [pt[j * 16:(j + 1) * 16] for j in range(len(pt) // 16)]
        if _xts_blocks(blocks, _expand_key_128(k1), _expand_key_128(k2), idx, decrypt=False) \
                != bytes.fromhex(ref):
            raise PkgError('XTS-AES-128 self-test failed (IEEE P1619/D16 vector, encrypt)')
        ct = bytes.fromhex(ref)
        cblocks = [ct[j * 16:(j + 1) * 16] for j in range(len(ct) // 16)]
        if _xts_blocks(cblocks, _expand_key_128(k1), _expand_key_128(k2), idx, decrypt=True) \
                != pt:
            raise PkgError('XTS-AES-128 self-test failed (IEEE P1619/D16 vector, decrypt)')
    # A 4096-byte sector (the PFS layout) must round-trip with a non-trivial tweak.
    buf = b'\xff' * XTS_SECTOR
    chunks = [buf[j * 16:(j + 1) * 16] for j in range(XTS_SECTOR // 16)]
    rk1, rk2 = _expand_key_128(bytes(15) + b'\x01'), _expand_key_128(bytes(15) + b'\x02')
    enc = _xts_blocks(chunks, rk1, rk2, 3, decrypt=False)
    dec = _xts_blocks([enc[j * 16:(j + 1) * 16] for j in range(XTS_SECTOR // 16)],
                      rk1, rk2, 3, decrypt=True)
    if dec != buf:
        raise PkgError('XTS-AES-128 self-test failed (4096-byte round-trip)')


# --------------------------------------------------------------------------- RSA
# The public key constants the community PKG tools ship (flatz/pkg_pfs_tool, ShadPKG's
# config.ini): each is (n, e, d); the entry payloads are RSA-2048 ciphertexts and only
# d and n are needed (a plain modular exponentiation).
_RSA_ENTRY_KEY_3 = (
    0xD212FC335F6DDB831609628B0356273782D477853529392D526B8C4C8CFB06C1845BE7D4F7BCD24E6245CD2ABBD77776453655273FB3F5F98EDA4BEFAA59AEB39BEA5498D206326A58312AE0D44F90B50A7DECF43A9C52672D99318E0C43E682FE0746E12E50D41F2D2F7ED908BA06B3BF2E203F4E3FFE44FFAA50435791699449158282E40F4C8D9D2CC95B1D64BF888BD4C594E76547841EE57910FB989347B97D8512A640982CF792BC951932EDE890560D65C1AA78C62E54FD5F54A1F67EE5E05F61C120B4B9B4330870E4DF8956ED012946775F8CB8A9F51E2EB3B9BFE009B78D28D4A6C3B81E1F07EBB4120B95B88530FDDC3913D07CDC8FEDF9C9A3C1,
    0x10001,
    0x32D903908FBDB08F572B285E0B8DB3EA5CD17EA890888CDD6A80BBB1DFC1F70DAA32F0B77CCB88800E8B64B0BE4CD60E9B8C1E2A64E1F35CD77601415E935C94FEDD4662C31B5AE2A0BC2DEBC3980AA7B7856970682B644AB31FCC7DDC7C26F477F65CF2AE5A442DD3AB16620419BAFB90FFE23050896ECB56B2EBC09116925E308EAEC7945DFD35E120F8AD3EBC08BFC036749FD5BB5208FD0666F37AB304F475295DE95FAA1030B20F5A1AC12AB3FECB21AD80EC8F20091CDBC55894C29CC6CE82653E5790BCA98B06B4F072F677DF9864F1ECFE372DBCAE8C08811FC3C9891AC742824B2EDC8E8D73CEB1CC01D90870873C4408EC498F815AE240FF77FC0D)
_RSA_EKPFS_KEY = (
    0xC6CF71E7E59AF0D12A2C458BF92A0EC143058BC37117801DCD497DDE359D259BA0D7A0F27D6C087EAA5502682B23C644B84418EB56CF16A24803C9E74F87EB3D30C31588BF20E79DFF770CDE1D241E63A94F8ABF5BBE601968333BFCED9F474E5FF8EACB3D00BD6701F92C6DC6AC1364E76714F3DC52696AB9832C4230131BB2D8A5020D79ED96B10DF8CC0CDF81954F035809570E80692EFEFF5277EA7528A8FBC9BEBF9FBBB7798E1805E180BD50349481D353C269A2D24CCF6CF4572C104A3FFB22FD8B97E2C95BA62BCDD61B6BDB687F4BC2A05034C005E58DEF2467FF9340CF2D62A2A050B1F13AA83DFD80D1F9B80522AFC8354590588EE33A7CBD3E27,
    0x10001,
    0x7F76CD0EE2D4DE051CC6D9A80E8DFA7BCA1EAA271A40F8F1228735DDDBFDEEF8C2BCBD01FB8BE23E63B2B1225C56496E11BE07440B9A2666D1492C8FD31BCFA4A1B8D1FBA49ED2212883098AF6A00BA3D60F9B6368CCBC0C4E145B27A4A9F42BB9B87BC0E651AD1D77D46BB9CE20D126667E5E9EA2E96B90F373B8528F4411030C1397393D132258D5438249DA6E7CA1C58CA5B009E0CE3DDFF49D3C9715E26AC72B3C509323DBBA4A226644AC78BB0E1A2743B57167AFF4AB48469373D042AB9363E56C9ADE5024C0237D99793F2207E0C148561BDF830912B42D456BC9C06885999079961AD7F54D1F3783404AEC3937A680927DC580C7D66FFE8A7989C6B1)


def _rsa_decrypt(blob, key):
    """RSA-2048 decrypt (a plain modular exponentiation; the PKCS#1 v1.5 padding is
    stripped: 00 01/02 <nonzero pad> 00 <message> -- the entry payloads are padded
    with 00 02, the type the community PKG tools use)."""
    n, _e, d = key
    m = pow(int.from_bytes(blob, 'big'), d, n)
    out = m.to_bytes((n.bit_length() + 7) // 8, 'big')
    if len(out) < 4 or out[0] != 0 or out[1] not in (1, 2):
        raise PkgError('RSA decrypt returned a non-PKCS#1 payload')
    i = 2
    while i < len(out) and out[i]:
        i += 1
    if i >= len(out):
        raise PkgError('RSA decrypt: no PKCS#1 separator')
    return out[i + 1:]


# --------------------------------------------------------------------------- SFO
def sfo_values(data):
    """The values of a param.sfo (a PSF file) as {key: value}. Mirrors prepare.sfo:
    \0PSF + size u32 + count u32, then (key u16, fmt u16, size u32, capacity u32,
    offset u32) rows, then the key and value string tables."""
    if not data:
        raise PkgError('param.sfo is empty (0 bytes)')
    try:
        magic, _size, keys, values, count = struct.unpack_from('<4sIIII', data, 0)
    except struct.error as error:
        raise PkgError(f'param.sfo is too short: {error}') from None
    if magic != b'\0PSF':
        raise PkgError('bad SFO signature (param.sfo)')
    result = {}
    for i in range(count):
        row = 20 + i * 16
        if row + 16 > len(data):
            raise PkgError(f'param.sfo: entry {i} points past the end of the file')
        key_off, fmt, size, _cap, off = struct.unpack_from('<HHIII', data, row)
        k_end = data.find(b'\0', keys + key_off)
        if k_end < 0 or off + size > len(data):
            raise PkgError(f'param.sfo: entry {i} points past the end of the file')
        k = data[keys + key_off:k_end].decode('utf-8', 'replace')
        v = data[values + off:values + off + size]
        if fmt == 0x404:
            if len(v) < 4:
                raise PkgError(f'param.sfo: entry {i} has a short int32 value')
            result[k] = struct.unpack('<I', v[:4])[0]
        else:
            result[k] = v.rstrip(b'\0').decode('utf-8', 'replace')
    return result


# --------------------------------------------------------------------------- PKG
class Pkg:
    """A .pkg file: the CNT/FSELF container, the entry table and the PFS image.

    Open with Pkg(path) -- or with an open binary file object; the expensive work
    (key derivation, the PFSC scan) happens once in pfs() and is cached, the PKG
    entry data is read lazily. The file is memory-mapped while the Pkg is open.
    """

    def __init__(self, source):
        self._file = None
        self._mm = None
        if isinstance(source, (str, os.PathLike)):
            self._file = open(os.fspath(source), 'rb')
            handle = self._file
            self.path = os.fspath(source)
        else:
            if not isinstance(source, (io.FileIO, io.BufferedReader)):
                raise PkgError('a .pkg path or an open binary file is needed')
            handle = source  # borrowed: the caller keeps it (close() releases only the mapping)
            self.path = getattr(source, 'name', '')
        try:
            mm = self._mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            if len(mm) < 0x440 or mm[0:4] != MAGIC_CNT:
                raise PkgError('not a PS4 .pkg (FSELF/CNT) file')
            self.entry_count, = struct.unpack_from('>I', mm, 0x10)
            self.entry_table, = struct.unpack_from('>I', mm, 0x18)
            self.body_offset, = struct.unpack_from('>Q', mm, 0x20)
            self.content_id = mm[0x40:0x50].rstrip(b'\0').decode('ascii', 'replace')
            self.pfs_count, = struct.unpack_from('>I', mm, 0x404)
            self.pfs_flags, = struct.unpack_from('>Q', mm, 0x408)
            self.pfs_offset, self.pfs_size = struct.unpack_from('>QQ', mm, 0x410)
            self.pfs_cache_size, = struct.unpack_from('>I', mm, 0x43C)
            if self.pfs_offset + min(self.pfs_size, 0x1000) > len(mm):
                raise PkgError(f'PFS image at {self.pfs_offset:#x} exceeds the file')
            # The entry table rows (32 bytes each, big-endian):
            # id u32, filename_offset u32, flags1 u32, flags2 u32, offset u32, size u32.
            if self.entry_table + self.entry_count * 32 > len(mm):
                raise PkgError('PKG entry table exceeds the file')
            self.entries = {}
            for i in range(self.entry_count):
                o = self.entry_table + i * 32
                entry_type, _fn, _f1, _f2, off, size = struct.unpack_from('>6I', mm, o)
                if off + size > len(mm):
                    raise PkgError(f'entry {entry_type:#x} points past the end of the file')
                self.entries[entry_type] = (off, size, mm[o:o + 32])
            if self.pfs_count != 1:
                raise PkgError(f'{self.pfs_count} PFS images: only single-image PKGs are supported')
        except Exception:
            self.close()
            raise
        self._pfs = None

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def entry(self, entry_type):
        """(offset, size) of a PKG entry, or None when it is not in the PKG."""
        row = self.entries.get(entry_type)
        return (row[0], row[1]) if row else None

    def entry_data(self, entry_type):
        """The bytes of a PKG entry (NPDRM entries 0x400..0x403 are AES-128-CBC
        decrypted with a dk3-derived key, the rest are stored as-is)."""
        row = self.entries.get(entry_type)
        if not row:
            raise PkgError(f'entry {entry_type:#x} is not in the PKG')
        data = self._mm[row[0]:row[0] + row[1]]
        if 0x400 <= entry_type <= 0x403:
            if not self.entry(0x20):
                raise PkgError(f'entry {entry_type:#x} is NPDRM but the PKG has no key entry (0x20)')
            dk3 = self._dk3()
            ivkey = hashlib.sha256(row[2] + dk3).digest()
            # Entries are zero-padded to a 16-byte boundary before decryption,
            # then truncated back to their size (npbind.dat is 532 bytes).
            padded = data.ljust((len(data) + 15) // 16 * 16, b'\0')
            data = _aes_cbc_decrypt(padded, ivkey[16:32], ivkey[0:16])[:len(data)]
        return data

    def sfo(self):
        """The param.sfo entry as {key: value} (raises PkgError without one)."""
        return sfo_values(self.entry_data(0x1000))

    def system_files(self):
        """{Sc0 relative name: bytes} of the PKG entries that belong in a game folder
        (param.sfo, the icons and pictures, the playgo and livearea files, the trophy
        files, the NPDRM files)."""
        files = {}
        for entry_type in sorted(self.entries):
            name = sc0_name(entry_type)
            if name:
                files[name] = self.entry_data(entry_type)
        return files

    # -- keys -----------------------------------------------------------------
    def _dk3(self):
        """dk3: entry 0x10 (entry_keys) holds 8 SHA256 digests, then 8 RSA-2048
        ciphertexts; the fourth (bytes 0x400..0x500) is decrypted with pkg_entry_key_3."""
        row = self.entry(0x10)
        if not row:
            raise PkgError('no entry_keys entry (0x10): the PFS keys cannot be derived')
        off10, size10 = row
        if size10 < 0x500:
            raise PkgError('the entry_keys entry (0x10) is too short')
        out = _rsa_decrypt(self._mm[off10 + 0x400:off10 + 0x500], _RSA_ENTRY_KEY_3)
        if len(out) != 32:
            raise PkgError(f'dk3 is {len(out)} bytes (expected 32)')
        return out

    def pfs_keys(self):
        """(K1 data key, K2 tweak key) for the PFS image's XTS-AES-128 payload:
        entry 0x20 (image_key) is AES-128-CBC decrypted with a dk3-derived key, the
        result is RSA-decrypted to ekpfs, and the keys are HMAC-SHA256(ekpfs,
        LE32(1) || seed) with the seed at PFS+0x370 (K2 is the first 16 bytes, K1
        the last 16)."""
        dk3 = self._dk3()
        row20 = self.entries.get(0x20)
        if not row20:
            raise PkgError('no image_key entry (0x20): the PFS keys cannot be derived')
        off20, size20 = row20[0], row20[1]
        if size20 < 0x100:
            raise PkgError('the image_key entry (0x20) is too short')
        # The key is keyed to the raw 32-byte entry table row (row20[2]), like
        # ShadPKG and pkg_pfs_tool derive it.
        ivkey = hashlib.sha256(row20[2] + dk3).digest()
        ekpfs = _rsa_decrypt(_aes_cbc_decrypt(self._mm[off20:off20 + 0x100],
                                               ivkey[16:32], ivkey[0:16]),
                             _RSA_EKPFS_KEY)
        if len(ekpfs) != 32:
            raise PkgError(f'ekpfs is {len(ekpfs)} bytes (expected 32)')
        if 0x380 > self.pfs_size:
            raise PkgError('the PFS image is too short for its crypt seed')
        seed = self._mm[self.pfs_offset + 0x370:self.pfs_offset + 0x380]
        enc = hmac.new(ekpfs, struct.pack('<I', 1) + seed, hashlib.sha256).digest()
        return enc[16:32], enc[0:16]

    # -- PFS ------------------------------------------------------------------
    def pfs(self):
        """Scan the PFS/PFSC image and return a PfsImage with its .files =
        [(relative posix path, size)] (paths relative to the uroot, the way the
        PS4 app images lay them out) and readers for them. Cached after the first call."""
        if self._pfs is not None:
            return self._pfs
        mm = self._mm
        pfs_off, pfs_size = self.pfs_offset, self.pfs_size
        if pfs_size < 0x4A0:
            raise PkgError('the PFS image is shorter than a PFS3 superblock')
        version, pfs_magic = struct.unpack_from('<QQ', mm, pfs_off)
        mode, = struct.unpack_from('<H', mm, pfs_off + 0x1C)
        block_size, = struct.unpack_from('<i', mm, pfs_off + 0x20)
        if pfs_magic != MAGIC_PFS or version != 1:
            raise PkgError(f'PFS image is not a PFS3 (version={version}, magic={pfs_magic:#x})')
        if not (0x2000 <= block_size <= 0x20000 and block_size & (block_size - 1) == 0):
            raise PkgError(f'PFS block size {block_size:#x}: only 0x2000..0x20000 '
                           'powers of two are supported')
        encrypted = bool(mode & MODE_ENCRYPTED)
        k1 = k2 = None
        if encrypted:
            k1, k2 = self.pfs_keys()
        reader = _PfsReader(mm, pfs_off, pfs_size, k1, k2, encrypted,
                            self.pfs_cache_size)
        reader.block_size = block_size
        # The PFSCHdr, when the image is compressed (PFSC, the layout of the shipped
        # PS4 game PKGs); a plain, uncompressed PFS3 image has no PFSCHdr and its
        # logical blocks sit at block i * block_size in the image.
        offset = reader.find_pfsc()
        if offset is not None:
            reader.compressed = True
            reader.pfsc_offset = offset
            hdr = reader.region(offset, 0x30)
            pfsc_magic, _u4, _u8, _blk = struct.unpack_from('<4i', hdr, 0)
            blk_sz2, blk_off, _data_start, data_len = struct.unpack_from('<4q', hdr, 0x10)
            if pfsc_magic != MAGIC_PFSC or not (0 < blk_sz2 <= 0x20000):
                raise PkgError('PFSCHdr is not intact')
            reader.block_size = blk_sz2
            reader.block_offsets = blk_off
            reader.data_len = data_len
            reader.load_sector_map(data_len // blk_sz2)
        elif encrypted:
            raise PkgError('no PFSCHdr in the PFS image (an encrypted, uncompressed PFS is not supported)')
        reader.load_inodes()
        reader.file_list()
        self._pfs = PfsImage(self, reader)
        return self._pfs


class PfsImage:
    """The scanned PFS of a Pkg: its file list and readers for the files (Image0/)."""

    def __init__(self, pkg, reader):
        self.pkg = pkg
        self.reader = reader
        self.files = reader._file_paths  # [(relative posix path, size)], PFS order
        self.encrypted = reader.encrypted

    def read_file(self, relative):
        """The bytes of one PFS file, by its relative path."""
        ino = self.reader._file_inos.get(relative)
        if ino is None:
            raise PkgError(f'no PFS file {relative!r}')
        return self.reader.read(ino)

    def iter_blocks(self, relative):
        """The 0x10000-byte blocks of one PFS file, streamed (for extraction)."""
        ino = self.reader._file_inos.get(relative)
        if ino is None:
            raise PkgError(f'no PFS file {relative!r}')
        yield from self.reader.iter_blocks(ino)

    def total_bytes(self):
        return sum(size for _name, size in self.files)


class _PfsReader:
    """XTS (or plain) access to one PFS image by PFS-image byte offset, and the PFSC
    logical-block reader (sector map + zlib) on top of it. XTS sector numbers are the
    absolute PFS-image offset / 4096 (the PFS image is sector-aligned in the PKG)."""

    def __init__(self, mm, pfs_off, pfs_size, k1, k2, encrypted, cache_size):
        self.mm = mm
        self.pfs_off = pfs_off
        self.pfs_size = pfs_size
        self.encrypted = encrypted
        self.cache_size = cache_size
        self.compressed = False
        self.pfsc_offset = None
        self.key32 = k1 + k2 if encrypted else None
        self._rk1 = _expand_key_128(k1) if encrypted else None
        self._rk2 = _expand_key_128(k2) if encrypted else None
        # cryptography rejects XTS with a duplicated key; guard K1 == K2 (rare).
        self._fast = Cipher is not None and encrypted and k1 != k2
        if encrypted and not self._fast:
            _selftest()
        self.block_size = BLOCK
        self.block_offsets = 0
        self.data_len = 0
        self.sector_map = None
        self.inodes = None
        self.children = None
        self._file_paths = None
        self._file_inos = None

    def region(self, img_off, nbytes):
        """`nbytes` at PFS-image offset `img_off`: XTS-decrypted (or plain) bytes."""
        if not 0 <= img_off and img_off + nbytes <= self.pfs_size:
            raise PkgError(f'PFS read at {img_off:#x}+{nbytes:#x} exceeds the image')
        if not self.encrypted:
            base = self.pfs_off + img_off
            return bytes(self.mm[base:base + nbytes])
        start = img_off & ~(XTS_SECTOR - 1)
        end = (img_off + nbytes + XTS_SECTOR - 1) & ~(XTS_SECTOR - 1)
        out = bytearray()
        for sec in range(start, end, XTS_SECTOR):
            raw = self.mm[self.pfs_off + sec:self.pfs_off + sec + XTS_SECTOR]
            if self._fast:
                out += _xts_sector_fast(raw, self.key32, sec // XTS_SECTOR)
            else:
                out += _xts_sector_pure(raw, self._rk1, self._rk2, sec // XTS_SECTOR)
        within = img_off - start
        return bytes(out[within:within + nbytes])

    def find_pfsc(self):
        """The PFS-image offset of the PFSCHdr (compressed PFS), or None for a plain
        PFS3 image: search the (XTS-decrypted) cached head for the magic on 0x10000
        boundaries, the way ShadPKG's GetPFSCOffset does (a plain image's head is
        scanned in place, without copying it)."""
        head_len = min(self.cache_size * 2, self.pfs_size) \
            if self.cache_size else min(0x200000, self.pfs_size)
        if self.encrypted and head_len < 0x30000:
            raise PkgError('PFS head too short to find the PFSCHdr')
        if self.encrypted:
            head = self.region(0, head_len)
            for i in range(0x20000, len(head), 0x10000):
                if struct.unpack_from('<I', head, i)[0] == MAGIC_PFSC:
                    return i
            return None
        for i in range(0x20000, head_len, 0x10000):
            if struct.unpack_from('<I', self.mm, self.pfs_off + i)[0] == MAGIC_PFSC:
                return i
        return None

    def logical_block(self, i):
        """The logical PFS block i (block_size bytes): through the PFSC sector map
        (zlib or raw) when compressed, a direct read at i * block_size otherwise."""
        if self.compressed:
            return self._pfsc_block(i)
        off = i * self.block_size
        if off + self.block_size > self.pfs_size:
            raise PkgError(f'PFS block {i} at {off:#x} exceeds the image')
        return self.region(off, self.block_size)

    def _pfsc_block(self, i):
        """The decompressed logical PFSC block i (block_size bytes)."""
        so = self.sector_map[i]
        span = self.sector_map[i + 1] - so
        base = self.pfsc_offset + so
        if span == 0:
            return bytes(self.block_size)
        data = self.region(base, span)
        if span == self.block_size:
            return data
        try:
            out = zlib.decompress(data)
        except zlib.error as error:
            raise PkgError(f'PFSC block {i}: zlib failed ({error})') from None
        if len(out) < self.block_size:
            raise PkgError(f'PFSC block {i}: {len(out)} bytes (expected {self.block_size:#x})')
        return out[:self.block_size]

    def load_sector_map(self, num_blocks):
        count = num_blocks + 1
        blob = self.region(self.pfsc_offset + self.block_offsets, count * 8)
        self.sector_map = list(struct.unpack_from(f'<{count}Q', blob, 0))
        for i in range(count - 1):
            if self.sector_map[i] > self.sector_map[i + 1]:
                raise PkgError(f'PFSC sector map not monotone at block {i}')
        if self.sector_map[-1] > self.data_len:
            raise PkgError('PFSC sector map extends past the data region')

    def load_inodes(self):
        blk0 = self.logical_block(0)
        self.nd_inodes, = struct.unpack_from('<I', blk0, 0x30)
        occupied = (self.nd_inodes * INODE_STRIDE + self.block_size - 1) // self.block_size
        inodes = []
        for bi in range(1, occupied + 1):
            blk = self.logical_block(bi)
            p = 0
            while p + INODE_STRIDE <= self.block_size:
                mode, = struct.unpack_from('<H', blk, p)
                if not mode:
                    break
                size, = struct.unpack_from('<q', blk, p + 8)
                blocks, loc = struct.unpack_from('<II', blk, p + 96)
                inodes.append((mode, size, blocks, loc))
                p += INODE_STRIDE
        if len(inodes) != self.nd_inodes:
            raise PkgError(f'{len(inodes)} inodes parsed but the superblock declares {self.nd_inodes}')
        self.inodes = inodes

    def file_list(self):
        """[(relative posix path, size)] of the PFS files. The PS4 app images keep
        everything under the root's uroot; the paths are relative to it, and the
        root's other children (like flat_path_table) are kept as-is."""
        inodes = self.inodes
        children = {}
        for n, (mode, size, blocks, loc) in enumerate(inodes):
            if not (mode & 0x4000) or not blocks:
                continue
            kids = []
            for j in range(blocks):
                blk = self.logical_block(loc + j)
                pos = 0
                while pos + 0x10 <= self.block_size:
                    ino, typ, namelen, entsize = struct.unpack_from('<4i', blk, pos)
                    if not ino or entsize < 0x10 or pos + 0x10 + namelen > self.block_size:
                        break
                    if typ in (2, 3):  # PFS_FILE / PFS_DIR
                        name = blk[pos + 0x10:pos + 0x10 + namelen].decode('utf-8', 'replace')
                        kids.append((name, ino, typ == 3))
                    pos += entsize
            children[n] = kids
        self.children = children
        dirs0 = [k for k in children.get(0, []) if k[2]]
        strip = 'uroot/' if len(dirs0) == 1 and dirs0[0][0] == 'uroot' else ''
        paths = []
        inos = {}
        seen = set()
        stack = [(0, '')]
        while stack:
            n, prefix = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for name, ino, is_dir in children.get(n, []):
                path = f'{prefix}/{name}'.lstrip('/')
                if strip and path.startswith(strip):
                    path = path[len(strip):]
                if is_dir:
                    stack.append((ino, path))
                elif ino < len(inodes):
                    paths.append((path, inodes[ino][1]))
                    inos[path] = ino
        self._file_paths = paths
        self._file_inos = inos
        return paths

    def read(self, ino):
        """The bytes of the PFS file with inode number `ino`."""
        out = bytearray()
        for blk in self.iter_blocks(ino):
            out += blk
        return bytes(out)

    def iter_blocks(self, ino):
        """The 0x10000-byte blocks of the PFS file with inode number `ino`
        (the last one trimmed to the file's declared size)."""
        if ino >= len(self.inodes):
            raise PkgError(f'inode {ino} is not in the PFS')
        mode, size, blocks, loc = self.inodes[ino]
        if not (mode & 0x8000) or blocks == 0:
            raise PkgError(f'inode {ino} is not a file')
        last = size - (blocks - 1) * self.block_size
        if not 0 < last <= self.block_size:
            raise PkgError(f'inode {ino}: the size {size} does not fit its {blocks} blocks')
        for j in range(blocks):
            blk = self.logical_block(loc + j)
            if j < blocks - 1:
                yield blk
            else:
                yield blk[:last]


# --------------------------------------------------------------------------- extract
class PkgProgress:
    """The progress report extract() passes to its callback: phase ('keys' or 'files'),
    a label (the current file in the 'files' phase), and done/total bytes (updated
    inside a file as well, every 64 MiB, not just between files)."""

    __slots__ = ('phase', 'label', 'done', 'total')

    def __init__(self, phase, label, done, total):
        self.phase = phase
        self.label = label
        self.done = done
        self.total = total

    def __repr__(self):
        return (f'PkgProgress({self.phase!r}, {self.label!r}, '
                f'{self.done}/{self.total})')


def atomic_write(dest, data):
    """Write `data` to `dest` whole-or-not-at-all (a .part file is renamed on success)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    try:
        with open(part, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def atomic_write_streaming(dest, chunks):
    """Write an iterable of byte chunks to `dest` whole-or-not-at-all."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    try:
        with open(part, 'wb') as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def extract(pkg, dest, on_progress=None, resume=True, pfs_sub='Image0', sc0_sub='Sc0',
            progress_step=64 * 2**20):
    """Write the PFS files under `dest/pfs_sub/` and the PKG entry files under
    `dest/sc0_sub/` from a .pkg (the game layout: Image0/ + the Sc0/ beside it; the
    launcher also unpacks DLC packages into one ADDCONT folder). `on_progress` (when
    given) gets a PkgProgress per step and, inside a PFS file, every `progress_step`
    bytes; with `resume`, files already at their full size are skipped, so an
    interrupted extraction continues where it left off.
    Returns (written, skipped) counts."""
    dest = pathlib.Path(dest)
    image, sc0 = dest / pfs_sub, dest / sc0_sub

    def report(phase, label, done, total):
        if on_progress:
            on_progress(PkgProgress(phase, label, done, total))

    # Phase 1: the keys and the PFS scan (small, but the PFSC head can be several MiB).
    report('keys', 'deriving the PFS keys', 0, 1)
    pfs = pkg.pfs()
    report('keys', 'PFS keys', 1, 1)
    sc0_files = pkg.system_files()
    # The big PFS files first: an interruption then has lost the least time.
    pfs_files = sorted(pfs.files, key=lambda item: -item[1])
    plan = [(pfs_sub, rel, size) for rel, size in pfs_files] + \
           [(sc0_sub, name, len(data)) for name, data in sorted(sc0_files.items())]
    total = sum(size for _t, _n, size in plan)
    # Fail on a too-small drive before the first byte is written (an interrupted
    # extraction on a full disk would otherwise stall near the end, or mid-file).
    need = 0
    for target, name, size in plan:
        out = (image if target == pfs_sub else sc0) / name
        if not (resume and out.is_file() and out.stat().st_size == size):
            need += size
    if need:
        folder = dest
        while not folder.exists() and folder.parent != folder:
            folder = folder.parent
        free = shutil.disk_usage(folder).free
        if free < need:
            raise PkgError(f'the destination drive has {free / 2**30:.1f} GiB free; '
                           f'unpacking needs {need / 2**30:.1f} GiB')
    done = written = skipped = 0
    report('files', 'extracting', 0, total)

    # Progress inside a file too: the game's ps4.cpk is most of the package, and a
    # per-file report would leave the bar still for its whole unpack.
    def streamed(name, base, target):
        off = last = 0
        for chunk in pfs.iter_blocks(name):
            off += len(chunk)
            if off - last >= progress_step:
                last = off
                report('files', f'{target}/{name}', base + off, total)
            yield chunk

    for target, name, size in plan:
        out = (image if target == pfs_sub else sc0) / name
        if resume and out.is_file() and out.stat().st_size == size:
            done += size
            skipped += 1
            report('files', f'{target}/{name}', done, total)
            continue
        if target == pfs_sub:
            atomic_write_streaming(out, streamed(name, done, target))
        else:
            atomic_write(out, sc0_files[name])
        done += size
        written += 1
        report('files', f'{target}/{name}', done, total)
    report('files', 'done', total, total)
    return written, skipped


def default_dest(pkg_path):
    """The extraction folder for a .pkg: beside it, named after the file (minus .pkg)."""
    p = pathlib.Path(pkg_path)
    if p.suffix.lower() == '.pkg':
        return p.parent / p.stem
    return p.parent / p.name


def main(argv=None):
    parser = argparse.ArgumentParser(description='extract a PS4 .pkg into Image0/ + Sc0/')
    parser.add_argument('pkg', help='the .pkg file')
    parser.add_argument('dest', nargs='?',
                        help='the destination folder (default: beside the .pkg)')
    parser.add_argument('--list', action='store_true',
                        help='list the files instead of extracting')
    parser.add_argument('--no-resume', action='store_true',
                        help='rewrite files even when complete')
    argv = list(sys.argv[1:]) if argv is None else argv
    args = parser.parse_args(argv)
    with Pkg(args.pkg) as pkg:
        pfs = pkg.pfs()
        sc0 = sorted(pkg.system_files().items())
        print(f'{args.pkg}: {pkg.content_id}')
        print(f'PFS image {pfs.total_bytes() / 1048576:.0f} MiB, '
              f'{"decrypted (XTS-AES-128)" if pfs.encrypted else "plain"}; Sc0 '
              f'{sum(len(d) for _n, d in sc0) / 1024:.0f} KiB')
        if args.list:
            for name, size in sorted(pfs.files):
                print(f'Image0/{name}  {size}')
            for name, data in sc0:
                print(f'Sc0/{name}  {len(data)}')
            return 0
        dest = args.dest or default_dest(args.pkg)
        mode = 'cryptography (C speed)' if pfs.reader._fast else \
               'pure-Python XTS (slow)' if pfs.encrypted else 'plain (no XTS)'
        print(f'extracting to {dest} with {mode}')

        def report(progress):
            if progress.phase == 'files':
                pct = 100.0 * progress.done / max(1, progress.total)
                print(f'\r  {progress.label}  {pct:5.1f}%  '
                      f'({progress.done / 1048576:.0f}/{progress.total / 1048576:.0f} MiB)',
                      end='', flush=True)
            else:
                print(f'  {progress.phase}: {progress.label}')
        written, skipped = extract(pkg, dest, report, resume=not args.no_resume)
        print(f'\ndone: {written} written, {skipped} already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())


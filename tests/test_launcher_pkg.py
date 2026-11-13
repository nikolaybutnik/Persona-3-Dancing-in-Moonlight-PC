"""The .pkg support: launcher/p3d_pkg.py (the CNT container, the PFS3/PFSC image,
the PKG entries) and the launcher's PKG rows (game and DLC).

The synthetic package is unsigned and plain (no XTS, no RSA, no NPDRM): a CNT
container with a param.sfo and an icon0.png entry, and a PFS image whose PFSC
region holds raw, zlib-compressed and zero blocks over a uroot tree that mirrors
the real game's layout (eboot.bin, data/, keystone, flat_path_table).
"""
import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
import zlib

import paths  # noqa: F401
from test_launcher_controls import launcher, sfo_bytes  # noqa: E402

sys.path.insert(0, str(paths.ROOT / 'launcher'))
import p3d_pkg  # noqa: E402

from content_profile import find_addons  # noqa: E402

BLOCK = 0x10000
EBOOT = b'\xE7' * BLOCK + b'\x11' * 0x1000           # 0x11000 bytes: two blocks, the last trimmed
CPK = b'C' * 0x500                                    # one zlib-compressed block
KEYSTONE = bytes(range(96))                          # one raw block, trimmed to 96
FPT = b'\x99' * 8                                     # flat_path_table: root level, one raw block
EXPECTED_FILES = {'data/cpk': CPK, 'eboot.bin': EBOOT, 'flat_path_table': FPT,
                  'keystone': KEYSTONE, 'zero.bin': b'\0' * 0x1000}


def _inode(mode, size=0, blocks=0, loc=0):
    """An 0xA8-byte PFSC inode row (mode, size, blocks, loc are the fields the reader uses)."""
    row = bytearray(0xA8)
    struct.pack_into('<H', row, 0, mode)
    struct.pack_into('<q', row, 8, size)
    struct.pack_into('<I', row, 0x60, blocks)
    struct.pack_into('<I', row, 0x64, loc)
    return bytes(row)


# The uroot tree: root(0) -> uroot(1) + flat_path_table(5); uroot -> eboot.bin(2), data(3),
# keystone(4), zero.bin(7); data -> cpk(6). The inode number is the table position.
INODES = [
    _inode(0x4000, blocks=1, loc=2),                # 0: root
    _inode(0x4000, blocks=1, loc=3),                # 1: uroot
    _inode(0x8000, size=0x11000, blocks=2, loc=5),  # 2: eboot.bin
    _inode(0x4000, blocks=1, loc=4),                # 3: data
    _inode(0x8000, size=96, blocks=1, loc=8),       # 4: keystone
    _inode(0x8000, size=8, blocks=1, loc=10),       # 5: flat_path_table
    _inode(0x8000, size=0x500, blocks=1, loc=7),    # 6: data/cpk
    _inode(0x8000, size=0x1000, blocks=1, loc=9),   # 7: zero.bin (a zero PFSC block)
]


def _dirent(name, ino, is_dir):
    """A packed PFS dirent: ino, type (2 file / 3 dir), namelen, entsize, name."""
    n = name.encode()
    return struct.pack('<4i', ino, 3 if is_dir else 2, len(n), 0x10 + len(n)) + n


def _dir_block(*entries):
    block = bytearray(BLOCK)
    pos = 0
    for e in entries:
        block[pos:pos + len(e)] = e
        pos += struct.unpack_from('<I', e, 0xC)[0]
    return bytes(block)


def _inode_table():
    block = bytearray(BLOCK)
    for i, row in enumerate(INODES):
        block[i * 0xA8:(i + 1) * 0xA8] = row
    return bytes(block)


def build_pfs_image():
    """A PFS3 image (mode 0x1, no encryption) with one PFSC region: blocks 0..6 and
    8 and 10 are raw, block 7 is zlib-compressed, block 9 is a zero block."""
    comp = zlib.compress(b'C' * BLOCK)
    l = len(comp)
    b0 = bytearray(BLOCK)
    struct.pack_into('<I', b0, 0x30, len(INODES))    # nd_inodes, where the PFS3 superblock keeps it
    blocks = [bytes(b0), _inode_table(),
              _dir_block(_dirent('uroot', 1, True), _dirent('flat_path_table', 5, False)),
              _dir_block(_dirent('eboot.bin', 2, False), _dirent('data', 3, True),
                         _dirent('keystone', 4, False), _dirent('zero.bin', 7, False)),
              _dir_block(_dirent('cpk', 6, False)),
              b'\xE7' * BLOCK, b'\x11' * BLOCK, comp,
              KEYSTONE + b'\0' * (BLOCK - 96), None, b'\x99' * BLOCK]
    # The sector map (relative to the PFSCHdr start): one entry per block plus the end.
    sector_map = ([0x10000 * i for i in range(1, 8)]
                  + [0x80000, 0x80000 + l, 0x90000 + l, 0x90000 + l, 0xA0000 + l])
    size = 0xC0000 + l
    img = bytearray(size)
    struct.pack_into('<QQ', img, 0, 1, 0x1332A0B)    # the PFS3 superblock: version 1, the magic
    struct.pack_into('<H', img, 0x1C, 0x1)           # mode: signed, not encrypted
    struct.pack_into('<i', img, 0x20, BLOCK)
    struct.pack_into('<4i4q', img, 0x20000, 0x43534650, 0, 0, BLOCK, BLOCK, 0x30, 0x10000, 11 * BLOCK)
    for i, v in enumerate(sector_map):
        struct.pack_into('<Q', img, 0x20030 + i * 8, v)
    for i in range(7):                                 # blocks 0..6 at relative 0x10000 * (i + 1)
        img[0x20000 + 0x10000 * (i + 1):0x20000 + 0x10000 * (i + 2)] = blocks[i]
    img[0x20000 + 0x80000:0x20000 + 0x80000 + l] = comp
    img[0x20000 + 0x80000 + l:0x20000 + 0x90000 + l] = blocks[8]
    img[0x20000 + 0x90000 + l:0x20000 + 0xA0000 + l] = blocks[10]
    return bytes(img)


def build_pfs_image_plain():
    """An uncompressed PFS3 image (mode 0x1, no PFSC region): block 0 is the superblock
    (nd_inodes at 0x30, as in the superblock), block 1 the inode table, and every
    logical block i sits at i * block_size in the image."""
    superblock = bytearray(BLOCK)
    struct.pack_into('<QQ', superblock, 0, 1, 0x1332A0B)  # version 1, the magic
    struct.pack_into('<H', superblock, 0x1C, 0x1)         # mode: signed, not encrypted
    struct.pack_into('<i', superblock, 0x20, BLOCK)
    struct.pack_into('<I', superblock, 0x30, len(INODES))  # nd_inodes
    blocks = {
        0: bytes(superblock),
        1: _inode_table(),
        2: _dir_block(_dirent('uroot', 1, True), _dirent('flat_path_table', 5, False)),
        3: _dir_block(_dirent('eboot.bin', 2, False), _dirent('data', 3, True),
                      _dirent('keystone', 4, False), _dirent('zero.bin', 7, False)),
        4: _dir_block(_dirent('cpk', 6, False)),
        5: b'\xE7' * BLOCK, 6: b'\x11' * BLOCK,
        7: CPK + b'\0' * (BLOCK - len(CPK)),
        8: KEYSTONE + b'\0' * (BLOCK - len(KEYSTONE)),
        9: b'\0' * BLOCK,
        10: b'\x99' * BLOCK,
    }
    img = bytearray(11 * BLOCK)
    for i, block in blocks.items():
        img[i * BLOCK:(i + 1) * BLOCK] = block
    return bytes(img)


def build_pkg(sfo, content_id='UP2611-CUSA12636', pfs=None):
    """A CNT container: a param.sfo (0x1000) and an icon0.png (0x1200) entry, then the PFS image."""
    icon = b'\x89PNG' + b'\x01' * 12
    pfs = build_pfs_image() if pfs is None else pfs
    out = bytearray(0x2000 + len(pfs))
    out[0:4] = b'\x7fCNT'
    struct.pack_into('>I', out, 0x10, 2)
    struct.pack_into('>I', out, 0x18, 0x1000)
    struct.pack_into('>Q', out, 0x20, 0x1040)
    out[0x40:0x50] = (content_id.encode() + b'\0' * 24)[:24]
    struct.pack_into('>I', out, 0x404, 1)
    struct.pack_into('>Q', out, 0x408, 0x3CC)
    struct.pack_into('>QQ', out, 0x410, 0x2000, len(pfs))
    struct.pack_into('>I', out, 0x43C, 0)
    rows = (struct.pack('>6I', 0x1000, 0, 0, 0, 0x1040, len(sfo)) + b'\0' * 8
            + struct.pack('>6I', 0x1200, 0, 0, 0, 0x1040 + len(sfo), len(icon)) + b'\0' * 8)
    out[0x1000:0x1040] = rows
    out[0x1040:0x1040 + len(sfo)] = sfo
    out[0x1040 + len(sfo):0x1040 + len(sfo) + len(icon)] = icon
    out[0x2000:0x2000 + len(pfs)] = pfs
    return bytes(out)


class PkgModuleTest(unittest.TestCase):
    """p3d_pkg.Pkg / p3d_pkg.extract on the synthetic package (plain PFS, no key derivation)."""

    def setUp(self):
        self.sfo = sfo_bytes(TITLE_ID='CUSA12636', APP_VER='01.00', TITLE='Persona 3: Dancing in Moonlight')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'test.pkg'
        self.path.write_bytes(build_pkg(self.sfo))

    def open(self):
        return p3d_pkg.Pkg(str(self.path))

    def test_container_entries_and_sfo(self):
        with self.open() as pkg:
            self.assertEqual(pkg.content_id, 'UP2611-CUSA12636')
            self.assertEqual(pkg.pfs_count, 1)
            self.assertEqual(pkg.entry(0x1000), (0x1040, len(self.sfo)))
            self.assertEqual(pkg.sfo()['TITLE_ID'], 'CUSA12636')
            self.assertEqual(pkg.sfo()['APP_VER'], '01.00')
            # The PKG entry files: Sc0/param.sfo + Sc0/icon0.png (a PKG without NPDRM has no npdrm/).
            sc0 = pkg.system_files()
            self.assertEqual(set(sc0), {'param.sfo', 'icon0.png'})
            self.assertEqual(sc0['param.sfo'], self.sfo)

    def test_rejects_a_non_pkg(self):
        bad = Path(self.temporary.name) / 'bad.pkg'
        bad.write_bytes(b'\x7fCNT' + b'\0' * 0x500)
        with self.assertRaises(p3d_pkg.PkgError):
            p3d_pkg.Pkg(str(bad))  # a container with no PFS image
        short = bytearray(self.path.read_bytes())
        struct.pack_into('>Q', short, 0x418, 0x100)  # a PFS image shorter than its superblock
        bad.write_bytes(bytes(short))
        with p3d_pkg.Pkg(str(bad)) as pkg:
            with self.assertRaises(p3d_pkg.PkgError):
                pkg.pfs()

    def test_pfs_file_list_is_relative_to_uroot(self):
        with self.open() as pkg:
            pfs = pkg.pfs()
            self.assertFalse(pfs.encrypted)  # plain image: no XTS, no key derivation
            self.assertEqual({name: size for name, size in pfs.files},
                              {name: len(data) for name, data in EXPECTED_FILES.items()})
            self.assertEqual(pfs.read_file('eboot.bin'), EBOOT)

    def test_extract_writes_image0_and_sc0(self):
        dest = Path(self.temporary.name) / 'out'
        with self.open() as pkg:
            written, skipped = p3d_pkg.extract(pkg, dest)
        self.assertEqual((written, skipped), (7, 0))
        for name, data in EXPECTED_FILES.items():
            self.assertEqual((dest / 'Image0' / name).read_bytes(), data, name)
        self.assertEqual((dest / 'Sc0' / 'param.sfo').read_bytes(), self.sfo)
        self.assertEqual((dest / 'Sc0' / 'icon0.png').read_bytes(), b'\x89PNG' + b'\x01' * 12)
        self.assertEqual(list(dest.rglob('*.part')), [])  # nothing half-written is left behind

    def test_extract_resumes_where_it_left_off(self):
        dest = Path(self.temporary.name) / 'out'
        with self.open() as pkg:
            self.assertEqual(p3d_pkg.extract(pkg, dest), (7, 0))
        (dest / 'Image0' / 'eboot.bin').write_bytes(b'cut short')
        (dest / 'Sc0' / 'param.sfo').write_bytes(b'')
        with self.open() as pkg:
            written, skipped = p3d_pkg.extract(pkg, dest, resume=True)
        self.assertEqual((written, skipped), (2, 5))
        self.assertEqual((dest / 'Image0' / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual((dest / 'Sc0' / 'param.sfo').read_bytes(), self.sfo)
        with self.open() as pkg:
            self.assertEqual(p3d_pkg.extract(pkg, dest, resume=False), (7, 0))  # forced rewrites

    def test_extract_into_custom_subfolders(self):
        dest = Path(self.temporary.name) / 'dlc'
        with self.open() as pkg:
            p3d_pkg.extract(pkg, dest, pfs_sub='stem/ADDCONT1', sc0_sub='stem/ADDCONT1/sce_sys')
        self.assertEqual((dest / 'stem/ADDCONT1' / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual((dest / 'stem/ADDCONT1/sce_sys' / 'param.sfo').read_bytes(), self.sfo)

    def test_progress_reports_inside_a_file(self):
        """The game's ps4.cpk is most of the package: a bar that only moved between
        files would sit at 0% while it unpacks, so extract() reports inside a file
        too (progress_step: one block here, 64 MiB by default)."""
        events = []
        dest = Path(self.temporary.name) / 'out'
        with self.open() as pkg:
            p3d_pkg.extract(pkg, dest, events.append, progress_step=65536)
        mid = [p for p in events if p.phase == 'files' and p.label == 'Image0/eboot.bin'
               and 0 < p.done < p.total]
        self.assertTrue(mid)
        self.assertEqual(mid[0].done, 65536)  # after the file's first block

    def test_plain_uncompressed_pfs_image_is_supported(self):
        """A PFS3 image without a PFSC region (an uncompressed dump, or a PKG editor's
        output): the same files, the blocks read straight at i * block_size."""
        plain = Path(self.temporary.name) / 'plain.pkg'
        plain.write_bytes(build_pkg(self.sfo, pfs=build_pfs_image_plain()))
        with p3d_pkg.Pkg(str(plain)) as pkg:
            pfs = pkg.pfs()
            self.assertFalse(pfs.encrypted)
            self.assertEqual({name: size for name, size in pfs.files},
                             {name: len(data) for name, data in EXPECTED_FILES.items()})
            self.assertEqual(pfs.read_file('eboot.bin'), EBOOT)
            self.assertEqual(pfs.read_file('data/cpk'), CPK)
            dest = Path(self.temporary.name) / 'out-plain'
            self.assertEqual(p3d_pkg.extract(pkg, dest), (7, 0))
        for name, data in EXPECTED_FILES.items():
            self.assertEqual((dest / 'Image0' / name).read_bytes(), data, name)

    def test_extract_refuses_a_full_drive(self):
        dest = Path(self.temporary.name) / 'out-full'
        with self.open() as pkg:
            with mock.patch.object(p3d_pkg.shutil, 'disk_usage', return_value=mock.Mock(free=0)):
                with self.assertRaises(p3d_pkg.PkgError):
                    p3d_pkg.extract(pkg, dest)
        self.assertFalse(dest.exists())  # nothing is written before the check

    def test_pkg_from_an_open_file_object(self):
        """Pkg() also takes an open binary file (the mapping borrows its handle)."""
        handle = open(self.path, 'rb')
        self.addCleanup(handle.close)
        pkg = p3d_pkg.Pkg(handle)
        self.addCleanup(pkg.close)
        self.assertEqual(pkg.pfs().read_file('eboot.bin'), EBOOT)
        pkg.close()
        self.assertFalse(handle.closed)  # a borrowed file object is not closed by Pkg


class LauncherPkgTest(unittest.TestCase):
    """The launcher's PKG rows: the title guard, the game unpack (and the game folder it
    sets) and the DLC unpack layout the add-on finder (scripts/content_profile.py) expects."""

    def setUp(self):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f'Tk display unavailable: {error}')
        self.root.withdraw()
        self.addCleanup(self.destroy_root)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / 'Sc0'  # an Sc0 beside the game, like a PKG tool's layout
        self.source.mkdir()
        (self.source / 'param.sfo').write_bytes(sfo_bytes(TITLE_ID='CUSA12636', APP_VER='01.02'))
        config = self.folder / 'config'
        config.mkdir()
        config_file = config / 'settings.json'
        config_file.write_text(json.dumps({**launcher.APP_DEFAULTS, 'game_dir': str(self.folder / 'Image0'),
                                           'ui_language': 'en', 'check_updates': False}), encoding='utf-8')
        for name, value in [('CONFIG_DIR', config), ('CONFIG_FILE', config_file), ('DATA_DIR', self.folder)]:
            patch = mock.patch.object(launcher, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        gpu = mock.patch.object(launcher, 'list_gpus', return_value=[])
        gpu.start()
        self.addCleanup(gpu.stop)
        self.app = launcher.Launcher(self.root, tk, ttk, filedialog, messagebox)
        self.app.messagebox = mock.Mock()
        self.root.update()
        self.sfo = sfo_bytes(TITLE_ID='CUSA12636', APP_VER='01.00', TITLE='Persona 3: Dancing in Moonlight')
        self.game_pkg = self.folder / 'P3D-Game-CUSA12636.pkg'
        self.game_pkg.write_bytes(build_pkg(self.sfo))
        self.dest = self.folder / 'extracted'
        self.dest.mkdir()
        self.app.filedialog = mock.Mock(askdirectory=mock.Mock(return_value=str(self.dest)))
        self.bar, self.percent, self.button = mock.Mock(), mock.Mock(), mock.Mock()

    def destroy_root(self):
        for callback in self.root.tk.call('after', 'info'):
            self.root.after_cancel(callback)
        self.root.destroy()

    def wait_for_unpack(self):
        deadline = time.monotonic() + 30
        while getattr(self.app, 'unpacking_pkg', False) and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(getattr(self.app, 'unpacking_pkg', False), 'the unpack did not finish')

    def test_game_pkg_unpack_sets_the_game_folder(self):
        self.app.var('game_pkg', 'app').set(str(self.game_pkg))
        self.app.extract_game_pkg(self.app.var('game_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        self.assertEqual((self.dest / 'Image0' / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual((self.dest / 'Sc0' / 'param.sfo').read_bytes(), self.sfo)
        self.assertEqual(self.app.var('game_dir', 'app').get(), str(self.dest / 'Image0'))
        text = self.app.messagebox.showinfo.call_args.args[1]
        self.assertIn('Unpacked', text)
        self.assertIn('Sc0', text)  # the offer of the copy into sce_sys at Play
        self.app.messagebox.showerror.assert_not_called()

    def test_another_titles_pkg_is_refused(self):
        wrong = self.folder / 'other.pkg'
        wrong.write_bytes(build_pkg(sfo_bytes(TITLE_ID='CUSA00001', TITLE='Other Game')))
        self.app.var('game_pkg', 'app').set(str(wrong))
        self.app.extract_game_pkg(self.app.var('game_pkg', 'app'), self.bar, self.percent, self.button)
        text = self.app.messagebox.showerror.call_args.args[1]
        self.assertIn('not Persona 3: Dancing in Moonlight', text)
        self.assertFalse((self.dest / 'Image0').exists())

    def test_no_pkg_chosen_is_named(self):
        self.app.extract_game_pkg(self.app.var('game_pkg', 'app'), self.bar, self.percent, self.button)
        self.assertIn('Choose the .pkg file first.', self.app.messagebox.showerror.call_args.args[1])

    def test_dlc_row_refuses_a_game_pkg(self):
        self.app.var('dlc_dir', 'app').set(str(self.dest))
        self.app.var('dlc_pkg', 'app').set(str(self.game_pkg))
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        text = self.app.messagebox.showerror.call_args.args[1]
        self.assertIn('not a DLC', text)

    def test_dlc_pkg_unpack_layout_is_found_by_the_addon_finder(self):
        label = 'ADDCONTDLCD0000C'
        dlc_sfo = sfo_bytes(CATEGORY='ac', TITLE_ID='CUSA12636', TITLE='P3D DLC',
                            CONTENT_ID=f'UP2611-{label}')
        dlc_pkg = self.folder / 'test-dlc.pkg'
        dlc_pkg.write_bytes(build_pkg(dlc_sfo))
        dlc_dir = self.folder / 'dlc'
        dlc_dir.mkdir()
        self.app.var('dlc_dir', 'app').set(str(dlc_dir))
        self.app.var('dlc_pkg', 'app').set(str(dlc_pkg))
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        base = dlc_dir / 'test-dlc' / label
        self.assertEqual((base / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual((base / 'sce_sys' / 'param.sfo').read_bytes(), dlc_sfo)
        addons = find_addons(str(dlc_dir), 'CUSA12636')
        self.assertEqual([a[0] for a in addons], [label])
        self.assertEqual(Path(addons[0][1]), base.resolve())
        self.app.messagebox.showerror.assert_not_called()

    def test_multiple_dlc_pkgs_are_all_unpacked_and_found(self):
        """One Unpack click for a ';'-joined selection: every package gets its own
        ADDCONT folder, and the add-on finder lists them all."""
        picked = []
        for label, title in (('ADDCONTDLCD0000C', 'P3D DLC One'), ('ADDCONTDLCD0000D', 'P3D DLC Two')):
            dlc_sfo = sfo_bytes(CATEGORY='ac', TITLE_ID='CUSA12636', TITLE=title,
                                CONTENT_ID=f'UP2611-{label}')
            pkg = self.folder / f'dlc-{label[-1]}.pkg'
            pkg.write_bytes(build_pkg(dlc_sfo))
            picked.append((label, pkg, dlc_sfo))
        dlc_dir = self.folder / 'dlc-multi'
        dlc_dir.mkdir()
        self.app.var('dlc_dir', 'app').set(str(dlc_dir))
        self.app.var('dlc_pkg', 'app').set(';'.join(str(pkg) for _label, pkg, _sfo in picked))
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        for label, pkg, dlc_sfo in picked:
            base = dlc_dir / pkg.stem / label
            self.assertEqual((base / 'eboot.bin').read_bytes(), EBOOT)
            self.assertEqual((base / 'sce_sys' / 'param.sfo').read_bytes(), dlc_sfo)
        addons = find_addons(str(dlc_dir), 'CUSA12636')
        self.assertEqual([a[0] for a in addons], sorted(label for label, _pkg, _sfo in picked))
        self.app.messagebox.showinfo.assert_called()
        self.app.messagebox.showerror.assert_not_called()

    def test_dlc_unpack_names_the_bad_package_and_installs_the_rest(self):
        """A base game package in the DLC row is named (with its own reason) and the
        valid add-on still unpacks."""
        label = 'ADDCONTDLCD0000C'
        dlc_sfo = sfo_bytes(CATEGORY='ac', TITLE_ID='CUSA12636', TITLE='P3D DLC',
                            CONTENT_ID=f'UP2611-{label}')
        good = self.folder / 'dlc-good.pkg'
        good.write_bytes(build_pkg(dlc_sfo))
        bad = self.folder / 'dlc-but-its-the-game.pkg'  # the right title, but not an add-on
        bad.write_bytes(build_pkg(self.sfo))
        dlc_dir = self.folder / 'dlc-mixed'
        dlc_dir.mkdir()
        self.app.var('dlc_dir', 'app').set(str(dlc_dir))
        self.app.var('dlc_pkg', 'app').set(f'{good};{bad}')
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        base = dlc_dir / 'dlc-good' / label
        self.assertEqual((base / 'eboot.bin').read_bytes(), EBOOT)
        text = self.app.messagebox.showerror.call_args.args[1]
        self.assertIn('Not unpacked:', text)
        self.assertIn('not a DLC', text)
        self.app.messagebox.showinfo.assert_not_called()

    def test_unset_dlc_folder_defaults_to_the_port_dlc_folder(self):
        """No stored DLC folder: Unpack unpacks into the port's own dlc folder (the first
        of the folders run.py looks for at Play) without asking, and names the
        destination in the result."""
        label = 'ADDCONTDLCD0000C'
        dlc_sfo = sfo_bytes(CATEGORY='ac', TITLE_ID='CUSA12636', TITLE='P3D DLC',
                            CONTENT_ID=f'UP2611-{label}')
        pkg = self.folder / 'test-dlc.pkg'
        pkg.write_bytes(build_pkg(dlc_sfo))
        self.app.var('dlc_pkg', 'app').set(str(pkg))
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        dlc_dir = self.folder / 'dlc'  # DATA_DIR (patched to self.folder) / dlc
        base = dlc_dir / 'test-dlc' / label
        self.assertEqual((base / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual((base / 'sce_sys' / 'param.sfo').read_bytes(), dlc_sfo)
        self.assertEqual(self.app.var('dlc_dir', 'app').get(), str(dlc_dir))  # remembered for next time
        self.app.filedialog.askdirectory.assert_not_called()  # no prompt: the default applied
        text = self.app.messagebox.showinfo.call_args.args[1]
        self.assertIn(str(dlc_dir), text)  # the destination was reported
        self.app.messagebox.showerror.assert_not_called()

    def test_a_missing_stored_dlc_folder_asks_again(self):
        """A stored DLC folder that no longer exists: the picker comes back (it was the
        folder that disappeared), and the choice is remembered."""
        label = 'ADDCONTDLCD0000C'
        dlc_sfo = sfo_bytes(CATEGORY='ac', TITLE_ID='CUSA12636', TITLE='P3D DLC',
                            CONTENT_ID=f'UP2611-{label}')
        pkg = self.folder / 'test-dlc.pkg'
        pkg.write_bytes(build_pkg(dlc_sfo))
        self.app.var('dlc_pkg', 'app').set(str(pkg))
        self.app.var('dlc_dir', 'app').set(str(self.folder / 'gone'))
        self.app.extract_dlc_pkg(self.app.var('dlc_pkg', 'app'), self.bar, self.percent, self.button)
        self.wait_for_unpack()
        self.app.filedialog.askdirectory.assert_called()
        base = self.dest / 'test-dlc' / label  # the mocked picker returns self.dest
        self.assertEqual((base / 'eboot.bin').read_bytes(), EBOOT)
        self.assertEqual(self.app.var('dlc_dir', 'app').get(), str(self.dest))

    def test_progress_bar_appears_only_while_unpacking(self):
        """The row's empty progress bar reads like a dead button: it is hidden at rest
        and the first progress event of an unpack shows it."""
        from tkinter import ttk

        bars = []

        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child, ttk.Progressbar):
                    bars.append(child)
                walk(child)

        walk(self.root)
        self.assertEqual(len(bars), 2)  # the game row and the DLC row
        for bar in bars:
            self.assertFalse(bar.winfo_ismapped())
        self.app.unpack_progress(bars[0], mock.Mock())(p3d_pkg.PkgProgress('keys', 'deriving', 0, 1))
        deadline = time.monotonic() + 5
        while not bars[0].winfo_ismapped() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(bars[0].winfo_ismapped())
        self.assertFalse(bars[1].winfo_ismapped())  # the other row's bar stays hidden


if __name__ == '__main__':
    unittest.main()

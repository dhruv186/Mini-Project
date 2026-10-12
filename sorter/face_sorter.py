import traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN
from .folder_utils import find_images, ensure_dir, copy_or_move, try_symlink_or_shortcut, create_link_to_file


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_and_detect(image_path, model="hog"):
    try:
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image, model=model)
        if not locations:
            return [], []
        encodings = face_recognition.face_encodings(image, locations)
        return locations, encodings
    except Exception:
        return [], []


def cluster_encodings(encodings, eps=0.6, min_samples=1):
    if len(encodings) == 0:
        return np.array([], dtype=int)
    X = np.vstack(encodings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(X)
    return labels


def face_distance(a, b):
    return np.linalg.norm(a - b)


def compute_folder_representative_embeddings(person_folder: Path, sample_limit=5, model="hog"):
    encs = []
    for p in find_images(person_folder):
        if len(encs) >= sample_limit:
            break
        _, e = load_and_detect(p, model=model)
        if e:
            encs.append(e[0])
    return encs


class FaceSorter:
    def __init__(self, src_dir, out_dir, eps=0.55, min_samples=1, model='cnn', move_files=False, match_thresh=0.55, logger=None):
        self.src_dir = Path(src_dir)
        self.out_dir = Path(out_dir)
        self.eps = eps
        self.min_samples = min_samples
        self.model = model
        self.move_files = move_files
        self.match_thresh = match_thresh
        self.logger = logger or (lambda *a, **k: None)

    def log(self, *args):
        try:
            self.logger(" ".join(map(str, args)))
        except Exception:
            pass

    def run(self):
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            images = list(find_images(self.src_dir))
            if not images:
                self.log("No images found.")
                return

            # detect faces & encodings
            face_records = []
            image_has_faces = {}
            for img in images:
                locs, encs = load_and_detect(img, model=self.model)
                if encs:
                    image_has_faces[img] = True
                    for i, e in enumerate(encs):
                        face_records.append({"image_path": img, "face_index": i, "encoding": e})
                else:
                    image_has_faces[img] = False

            if not face_records:
                nf = self.out_dir / 'no_faces'
                nf.mkdir(exist_ok=True)
                for img in images:
                    copy_or_move(Path(img), nf / Path(img).name, move=self.move_files)
                self.log("Moved/Copied all to no_faces")
                return

            encodings = [r['encoding'] for r in face_records]
            labels = cluster_encodings(encodings, eps=self.eps, min_samples=self.min_samples)
            clusters = defaultdict(list)
            for rec, lbl in zip(face_records, labels):
                clusters[lbl].append(rec)

            # existing person folders
            existing_person_folders = {p.name: p for p in self.out_dir.iterdir() if p.is_dir() and p.name.startswith('person_')}
            person_reps = {}
            for name, folder in existing_person_folders.items():
                reps = compute_folder_representative_embeddings(folder, sample_limit=6, model=self.model)
                if reps:
                    person_reps[name] = reps

            def find_matching_person(cluster_encs):
                centroid = np.mean(np.vstack(cluster_encs), axis=0)
                best_name = None
                best_dist = float('inf')
                for name, rep_list in person_reps.items():
                    dists = [face_distance(centroid, rep) for rep in rep_list]
                    min_d = min(dists) if dists else float('inf')
                    if min_d < best_dist:
                        best_dist = min_d
                        best_name = name
                if best_name and best_dist <= self.match_thresh:
                    return best_name, best_dist
                return None, None

            # prepare new person indices
            existing_nums = [int(p.name.split('_')[1]) for p in self.out_dir.iterdir() if p.is_dir() and p.name.startswith('person_') and len(p.name.split('_'))>1 and p.name.split('_')[1].isdigit()]
            next_index = max(existing_nums) + 1 if existing_nums else 0

            cluster_folder_map = {}
            for lbl, recs in clusters.items():
                if lbl == -1:
                    continue
                folder = None
                match_name, dist = find_matching_person([r['encoding'] for r in recs])
                if match_name:
                    folder = self.out_dir / match_name
                    self.log(f"Cluster {lbl} matched existing {match_name} (dist={dist:.3f})")
                else:
                    folder = self.out_dir / f"person_{next_index:03d}"
                    folder.mkdir(exist_ok=True)
                    self.log(f"Cluster {lbl} -> new folder {folder.name}")
                    next_index += 1
                cluster_folder_map[lbl] = folder

            unknown_dir = self.out_dir / 'unknown'
            unknown_dir.mkdir(exist_ok=True)
            no_faces_dir = self.out_dir / 'no_faces'
            no_faces_dir.mkdir(exist_ok=True)
            group_dir = self.out_dir / 'group_photos'
            group_dir.mkdir(exist_ok=True)

            processed_group_targets = {}

            # process clusters
            for lbl, recs in clusters.items():
                if lbl == -1:
                    for rec in recs:
                        src = Path(rec['image_path'])
                        copy_or_move(src, unknown_dir / src.name, move=self.move_files)
                    continue

                folder = cluster_folder_map[lbl]
                for rec in recs:
                    src = Path(rec['image_path'])
                    image_face_count = sum(1 for r2 in face_records if r2['image_path'] == rec['image_path'])
                    if image_face_count <= 1:
                        dst = folder / src.name
                        if dst.exists():
                            base, ext = src.stem, src.suffix
                            i = 1
                            while (folder / f"{base}_copy{i}{ext}").exists():
                                i += 1
                            dst = folder / f"{base}_copy{i}{ext}"
                        copy_or_move(src, dst, move=self.move_files)
                    else:
                        if src not in processed_group_targets:
                            dst_group = group_dir / src.name
                            if dst_group.exists():
                                base, ext = src.stem, src.suffix
                                i = 1
                                while (group_dir / f"{base}_copy{i}{ext}").exists():
                                    i += 1
                                dst_group = group_dir / f"{base}_copy{i}{ext}"
                            copy_or_move(src, dst_group, move=self.move_files)
                            processed_group_targets[src] = dst_group
                        else:
                            dst_group = processed_group_targets[src]

                        link_name = folder / dst_group.name
                        if not link_name.exists():
                            try:
                                create_link_to_file(dst_group, link_name)
                            except Exception as e:
                                # fallback: copy (or log) - keep behavior explicit
                                self.log(f"[WARN] Could not create link for {dst_group} -> {link_name}: {e}. Falling back to copy.")
                                # create a copy only if you must; otherwise leave it un-copied
                                copy_or_move(dst_group, link_name, move=self.move_files)


            # copy/move no-face images
            for img_path, has_face in image_has_faces.items():
                if not has_face:
                    src = Path(img_path)
                    copy_or_move(src, no_faces_dir / src.name, move=self.move_files)

            self.log('Done')
        except Exception as e:
            self.log('Error in FaceSorter: ' + str(e))
            self.log(traceback.format_exc())
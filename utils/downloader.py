import os


from utils.mpr import MultipleProcessRunner
from tqdm import tqdm


class Downloader(MultipleProcessRunner):
    """
        Download files that has unified resource locator
    """
    
    def __init__(self, base_url, save_path, overwrite=False, skip_error_info=False, **kwargs):
        """

        Args:
            base_url: Unified Resource Locator of pdb file
            save_path: Unified Resource Locator of saving path
            overwrite: whether overwrite existing files
        """
        super().__init__(**kwargs)
        
        self.base_url = base_url
        self.save_path = save_path
        self.overwrite = overwrite
        self.skip_error_info = skip_error_info
        
        if not overwrite:
            # remove existing files in data
            self.data = [uniprot for uniprot in tqdm(self.data, desc="Filtering out existing files...")
                         if not os.path.exists(self.save_path.format(uniprot))]
    
    def _aggregate(self, final_path: str, sub_paths):
        pass

    def _target_static(self, process_id, data, sub_path, *args):
        for i, uniprot in enumerate(data):
            url = self.base_url.format(uniprot)
            save_path = self.save_path.format(uniprot)
            
            # shell cmd to download files
            wget = f"wget -q -o /dev/null {url} -O {save_path}"

            rm = f"rm {save_path}"
            err = f"echo 'Error: {url} cannot be downloaded!'"
            if self.skip_error_info:
                err += ">/dev/null"
                
            os.system(f"{wget} || ({rm} && {err})")

            self.terminal_progress_bar(process_id, i + 1, len(data), f"Process{process_id} Downloading files...")
    
    def run(self):
        """
            Run this function to download files
        """
        super().run()
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    # Clear empty files in specific directory
    def clear_empty_files(path):
        cnt = 0
        for file in tqdm(os.listdir(path), desc="Clearing empty files..."):
            if os.path.getsize(os.path.join(path, file)) == 0:
                os.remove(os.path.join(path, file))
                cnt += 1
        print(f"Removed {cnt} empty files")
        return cnt


class AlphaDBDownloader(Downloader):
    """
        Download files from AlphaFold2 database
    """
    def __init__(self, uniprot_ids, type: str, save_dir: str, **kwargs):
        """
        
        Args:
            uniprots: Uniprot ids
            type: Which type of files to download. Must be one of ['pdb', 'mmcif', 'plddt', "pae"]
            save_dir: Saving directory
            **kwargs:
        """

        url_dict = {
            "pdb": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb",
            "mmcif": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.cif",
            "plddt": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-confidence_v4.json",
            "pae": "https://alphafold.ebi.ac.uk/files/AF-{}-F1-predicted_aligned_error_v4.json"
        }
        
        save_dict = {
            "pdb": "{}.pdb",
            "mmcif": "{}.cif",
            "plddt": "{}.json",
            "pae": "{}.json"
        }
        base_url = url_dict[type]
        save_path = os.path.join(save_dir, save_dict[type])
        
        super().__init__(data=uniprot_ids, base_url=base_url, save_path=save_path, **kwargs)


class PDBDownloader(Downloader):
    """
        Download files from PDB
    """
    def __init__(self, pdb_ids, type: str, save_dir: str, **kwargs):
        """
        
        Args:
            pdb_ids: PDB ids
            type: Which type of files to download. Must be one of ['pdb', 'mmcif']
            save_dir: Saving directory
        """
        
        url_dict = {
            "pdb": "https://files.rcsb.org/download/{}.pdb",
            "mmcif": "https://files.rcsb.org/download/{}.cif"
        }
        
        save_dict = {
            "pdb": "{}.pdb",
            "mmcif": "{}.cif"
        }
        
        base_url = url_dict[type]
        save_path = os.path.join(save_dir, save_dict[type])
        
        super().__init__(data=pdb_ids, base_url=base_url, save_path=save_path, **kwargs)


class CATHDownloader(Downloader):
    def __init__(self, cath_ids, save_dir, **kwargs):
        """
            Download files from CATH
        Args:
            cath_ids: CATH ids
            save_dir: Saving directory
        """
        
        url = "http://www.cathdb.info/version/v4_3_0/api/rest/id/{}.pdb"
        save_path = os.path.join(save_dir, "{}.pdb")
        
        super().__init__(data=cath_ids, base_url=url, save_path=save_path, **kwargs)
    

def download_pdb(pdb_id: str, format: str, save_path: str):
    """
    Download pdb file from PDB
    Args:
        pdb_id: PDB id
        format: File , must be one of ['pdb', 'cif']
        save_path: Saving path
    """
    
    url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
    wget = f"wget -q -o /dev/null {url} -O {save_path}"
    rm = f"rm {save_path}"
    err = f"echo 'Error: {url} cannot be downloaded!'"
    os.system(f"{wget} || ({rm} && {err})")
    

def download_af2(uniprot_id: str, format: str, save_path: str):
    """
    Download files from AlphaFold2 database
    Args:
        uniprot_id: Uniprot id
        format: File format, must be one of ['pdb', 'cif', 'plddt', 'pae']
        save_path: Saving path
    """
    
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.{format}"
    wget = f"wget -q -o /dev/null {url} -O {save_path}"
    rm = f"rm {save_path}"
    err = f"echo 'Error: {url} cannot be downloaded!'"
    os.system(f"{wget} || ({rm} && {err})")
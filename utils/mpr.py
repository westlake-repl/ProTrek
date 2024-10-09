import abc
import os
import time
import sys


from tqdm import tqdm
from math import ceil


class MultipleProcessRunner:
    """
    Abstarct class for running tasks with multiple process
    There are three abstract methods that should be implemented:
        1. __len__() : return the length of data
        2. _target() : target function for each process
        3. _aggregate() : aggregate results from each process
    """
    
    def __init__(self,
                 data,
                 save_path=None,
                 n_process=1,
                 verbose=True,
                 total_only=True,
                 log_step=1,
                 start_method='fork',
                 split_strategy="static"):
        """
        Args:
            data     : data to be processed that can be sliced
            
            path     : final output path
            
            n_process: number of process
            
            verbose  : if True, display progress bar
            
            total_only: If True, only total progress bar is displayed
            
            log_step : For total progress bar, Next log will be printed when ``current iteration`` - ``last log
                       iteration`` >= log_step
            
            start_method: start method for multiprocessing
            
            split_strategy: method to split data, can be 'queue', 'static'. If 'queue', data will be put into a
                            queue and each process will get data from the queue. If 'static', data will be split
                            into n_process parts and each process will get one part.
        """
        self.data = data
        self.save_path = save_path
        self.n_process = n_process
        self.verbose = verbose
        self.total_only = total_only
        self.log_step = log_step
        self.start_method = start_method
        self.split_strategy = split_strategy
        
        assert self.split_strategy in ["queue", "static"], f"Split strategy must be 'queue' or 'static', but got {self.split_strategy}"
        
        # get terminal width to format output
        try:
            self.terminal_y = os.get_terminal_size()[0]

        except Exception as e:
            print(e)
            print("Can't get terminal size, set terminal_y = None")
            self.terminal_y = None
    
    def _s2hms(self, seconds: float):
        """
        convert second format of time into hour:minute:second format

        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        
        return "%02d:%02d:%02d" % (h, m, s)

    def _display_time(self, st_time, now, total):
        ed_time = time.time()
        running_time = ed_time - st_time
        rest_time = running_time * (total - now) / now
        iter_sec = f"{now / running_time:.2f}it/s" if now > running_time else f"{running_time / now:.2f}s/it"
        
        return f' [{self._s2hms(running_time)} < {self._s2hms(rest_time)}, {iter_sec}]'

    def _display_bar(self, now, total, length):
        now = now if now <= total else total
        num = now * length // total
        progress_bar = '[' + '#' * num + '_' * (length - num) + ']'
        return progress_bar

    def _display_all(self, now, total, desc, st_time):
        # make a progress bar
        length = 50
        progress_bar = self._display_bar(now, total, length)
        time_display = self._display_time(st_time, now, total)

        display = f'{desc}{progress_bar} {int(now / total * 100):02d}% {now}/{total}{time_display}'
        
        # Clean a line
        width = self.terminal_y if self.terminal_y is not None else 100
        num_space = width - len(display)
        if num_space > 0:
            display += ' ' * num_space
        else:
            length += num_space
            progress_bar = self._display_bar(now, total, length)
            display = f'{desc}{progress_bar} {int(now / total * 100):02d}% {now}/{total}{time_display}'

        # Set color
        display = f"\033[31m{display}\033[0m"
        
        return display
    
    # Print progress bar at specific position in terminal
    def terminal_progress_bar(self,
                              process_id: int,
                              now: int,
                              total: int,
                              desc: str = ''):
        """

        Args:
            process_id: process id
            now: now iteration number
            total: total iteration number
            desc: description

        """
        st_time = self.process_st_time[process_id]
        
        # Aggregate total information
        self.counts[process_id] = now
        self._total_display(self.process_st_time["total"])
        
        if not self.total_only:
            process_display = self._display_all(now, total, desc, st_time)
            if self.terminal_y is not None:
                sys.stdout.write(f"\x1b7\x1b[{process_id + 1};{0}f{process_display}\x1b8")
                sys.stdout.flush()
            else:
                print(f"\x1b7\x1b[{process_id + 1};{0}f{process_display}\x1b8", flush=True)

    # Print global information
    def _total_display(self, st_time):
        if self.total_display_callable.value == 1:
            self.total_display_callable.value = 0
            
            cnt = sum([self.counts[i] for i in range(self.n_process)])
            if cnt - self.last_cnt.value >= self.log_step:
                total_display = self._display_all(cnt, self.__len__(), f"Total: ", st_time)
                self.last_cnt.value = cnt
    
                x = self.n_process + 1 if not self.total_only else 0
                # if self.terminal_y is not None:
                #     sys.stdout.write(f"\x1b7\x1b[{x};{0}f{total_display}\x1b8")
                #     sys.stdout.flush()
                # else:
                #     print(f"\x1b7\x1b[{x};{0}f{total_display}\x1b8", flush=True)
                print(f"\r\x1b7\x1b[{x};{0}f{total_display}\x1b8", flush=True, end="")
            
            self.total_display_callable.value = 1

    def run(self):
        """
        The function is used to run a multi-process task
        Returns: return the result of function '_aggregate()'
        """
        
        if self.split_strategy == "static":
            return self.run_static()
        
        elif self.split_strategy == "queue":
            return self.run_queue()
    
    def run_static(self):
        """
        Running multi-process task with static data splits
        """
        
        import multiprocess as mp
        mp.set_start_method(self.start_method, force=True)
        
        # total number of data that is already processed
        self.counts = mp.Manager().dict({i: 0 for i in range(self.n_process)})
        
        # record start time for each process
        self.process_st_time = {"total": time.time()}
        
        # set a lock to call total number display
        self.total_display_callable = mp.Value('d', 1)
        
        # Save last log iteration number
        self.last_cnt = mp.Value('d', 0)
        
        num_per_process = ceil(self.__len__() / self.n_process)
        
        if self.save_path is not None:
            file_name, suffix = os.path.splitext(self.save_path)
        
        process_list = []
        sub_paths = []
        for i in range(self.n_process):
            st = i * num_per_process
            ed = st + num_per_process
            
            # construct slice and sub path for sub process
            data_slice = self.data[st: ed]
            
            sub_path = None
            # Create a directory to save sub-results
            if self.save_path is not None:
                save_dir = f"{file_name}{suffix}_temp"
                os.makedirs(save_dir, exist_ok=True)
                sub_path = f"{save_dir}/temp_{i}{suffix}"
            
            # construct sub process
            input_args = (i, data_slice, sub_path)
            self.process_st_time[i] = time.time()
            p = mp.Process(target=self._target_static, args=input_args)
            p.start()
            
            process_list.append(p)
            sub_paths.append(sub_path)
        
        for p in process_list:
            p.join()
        
        # aggregate results and remove temporary directory
        results = self._aggregate(self.save_path, sub_paths)
        if self.save_path is not None:
            save_dir = f"{file_name}{suffix}_temp"
            os.rmdir(save_dir)
        
        return results
    
    def run_queue(self):
        """
        Running multi-process task with shared queue
        """
        
        import multiprocess as mp
        mp.set_start_method(self.start_method, force=True)
        
        # total number of data that is already processed
        self.counts = mp.Manager().dict({i: 0 for i in range(self.n_process)})
        
        # Initialize a queue to input data
        self.q = mp.Queue(self.__len__())
        iterator = tqdm(self.data, "Input data to queue") if self.verbose else self.data
        for d in iterator:
            self.q.put(d)
        
        # record start time for each processs
        self.process_st_time = {"total": time.time()}
        
        # set a lock to call total number display
        self.total_display_callable = mp.Value('d', 1)
        
        # Save last log iteration number
        self.last_cnt = mp.Value('d', 0)
        
        if self.save_path is not None:
            file_name, suffix = os.path.splitext(self.save_path)
        
        process_list = []
        sub_paths = []
        for i in range(self.n_process):
            sub_path = None
            # Create a directory to save sub-results
            if self.save_path is not None:
                save_dir = f"{file_name}{suffix}_temp"
                os.makedirs(save_dir, exist_ok=True)
                sub_path = f"{save_dir}/temp_{i}{suffix}"
            
            # construct sub process
            input_args = (i, sub_path)
            self.process_st_time[i] = time.time()
            p = mp.Process(target=self._target_queue, args=input_args)
            p.start()
            
            process_list.append(p)
            sub_paths.append(sub_path)
        
        for p in process_list:
            p.join()
        
        # aggregate results and remove temporary directory
        results = self._aggregate(self.save_path, sub_paths)
        if self.save_path is not None:
            save_dir = f"{file_name}{suffix}_temp"
            os.rmdir(save_dir)
        
        return results
    
    @abc.abstractmethod
    def _aggregate(self, final_path: str, sub_paths):
        """
        This function is used to aggregate results from sub processes into a file

        Args:
            final_path: path to save final results
            sub_paths : list of sub paths

        Returns: None or desirable results specified by user

        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _target_static(self, process_id, data, sub_path):
        """
        The main body to operate data in one process. This function is used when split_strategy is 'static'.

        Args:
            i       : process id
            data    : data slice
            sub_path: sub path to save results
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def _target_queue(self, process_id, sub_path):
        """
        The main body to operate data in one process. This function is used when split_strategy is 'queue'.

        Args:
            i       : process id
            sub_path: sub path to save results
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError
    

class MultipleProcessRunnerSimplifier(MultipleProcessRunner):
    """
    A simplified version of MultipleProcessRunner.
    User only need to implement the function 'do', then it will be automatically executed
    in every iteration after call the function 'run'.
    If 'save_path' is specified, it will open a file in the 'sub_path' into which
    user can write results, and results will be aggregated into 'save_path'.
    
    The procedure would be like:
        ...
        with open(sub_path, 'w') as w:
            for i, d in enumerate(data):
                self.do(process_id, i, d, w) # You can write results into the file.
                ...
        
    The 'do' function should be like:
        def do(process_id, idx, data, writer):
            ...
    
    If 'save_path' is None, the argument 'writer' will be set to None.
    
    """
    
    def __init__(self, data, do, return_results=False, **kwargs):

        super().__init__(data=data, **kwargs)
        self.do = do
        self.return_results = return_results
    
    def run(self):
        self.start_time = time.time()
        return super().run()
    
    def _aggregate(self, final_path: str, sub_paths):
        results = []
            
        w = open(final_path, 'w') if final_path is not None else None
        
        if self.verbose:
            iterator = tqdm(enumerate(sub_paths), "Aggregating results...")
        else:
            iterator = enumerate(sub_paths)
        
        for i, sub_path in iterator:
            if sub_path is None and self.return_results:
                sub_path = f"MultipleProcessRunnerSimplifier_{self.start_time}_{i}.tmp"
            
            if sub_path is not None:
                with open(sub_path, 'r') as r:
                    for line in r:
                        if w is not None:
                            w.write(line)

                        if self.return_results:
                            results.append(line[:-1])
            
                os.remove(sub_path)
        
        return results
                
    def _target_static(self, process_id, data, sub_path):
        if sub_path is None and self.return_results:
            sub_path = f"MultipleProcessRunnerSimplifier_{self.start_time}_{process_id}.tmp"
            
        w = open(sub_path, 'w') if sub_path is not None else None
        for i, d in enumerate(data):
            self.do(process_id, i, d, w)
            if self.verbose:
                self.terminal_progress_bar(process_id, i + 1, len(data), f"Process{process_id} running...")
        
        if w is not None:
            w.close()
    
    def _target_queue(self, process_id, sub_path):
        if sub_path is None and self.return_results:
            sub_path = f"MultipleProcessRunnerSimplifier_{self.start_time}_{process_id}.tmp"
        
        w = open(sub_path, 'w') if sub_path is not None else None
        i = 0
        while not self.q.empty():
            try:
                d = self.q.get(timeout=1)
            except Exception:
                continue
                
            self.do(process_id, i, d, w)
            if self.verbose:
                self.terminal_progress_bar(process_id, i + 1, self.__len__(), f"Process{process_id} running...")
            
            i += 1
            
        if w is not None:
            w.close()
        
    def __len__(self):
        return len(self.data)
    
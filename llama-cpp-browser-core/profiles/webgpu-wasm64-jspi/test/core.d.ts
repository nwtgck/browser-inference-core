// TypeScript bindings for emscripten-generated code.  Automatically generated at compile time.
declare var RuntimeExports: {
    /** @type {!Uint8Array} */
    HEAPU8: Uint8Array;
    FS: {
        root: null;
        mounts: never[];
        devices: {};
        streams: never[];
        nextInode: number;
        nameTable: null;
        currentPath: string;
        initialized: boolean;
        ignorePermissions: boolean;
        filesystems: null;
        syncFSRequests: number;
        ErrnoError: {
            new (errno: any): {
                cause?: unknown;
                name: string;
                errno: any;
                code: string | undefined;
                message: string;
                stack?: string;
            };
        };
        FSStream: {
            new (): {
                shared: {};
                get object(): any;
                set object(val: any);
                node: any;
                get isRead(): boolean;
                get isWrite(): boolean;
                get isAppend(): number;
                flags: any;
                position: any;
            };
        };
        FSNode: {
            new (parent: any, name: any, mode: any, rdev: any): {
                node_ops: {};
                stream_ops: {};
                readMode: number;
                writeMode: number;
                mounted: null;
                parent: any;
                mount: any;
                id: number;
                name: any;
                mode: any;
                rdev: any;
                atime: number;
                mtime: number;
                ctime: number;
                get read(): boolean;
                set read(val: boolean);
                get write(): boolean;
                set write(val: boolean);
                readonly isFolder: any;
                readonly isDevice: any;
                addListener(cb: any, exclusive?: boolean): {
                    listeners: any;
                    entry: {
                        cb: any;
                        exclusive: boolean;
                    };
                };
                notifyListeners(flags: any): void;
                exclTurn: any;
            };
        };
        lookupPath(path: any, opts?: {}): {
            path: string;
            node?: undefined;
        } | {
            path: string;
            node: any;
        };
        getPath(node: any): any;
        hashName(parentid: any, name: any): number;
        hashAddNode(node: any): void;
        hashRemoveNode(node: any): void;
        lookupNode(parent: any, name: any): any;
        createNode(parent: any, name: any, mode: any, rdev: any): any;
        destroyNode(node: any): void;
        isRoot(node: any): boolean;
        isMountpoint(node: any): boolean;
        isFile(mode: any): boolean;
        isDir(mode: any): boolean;
        isLink(mode: any): boolean;
        isChrdev(mode: any): boolean;
        isBlkdev(mode: any): boolean;
        isFIFO(mode: any): boolean;
        isSocket(mode: any): boolean;
        flagsToPermissionString(flag: any): string;
        nodePermissions(node: any, perms: any): 0 | 2;
        mayLookup(dir: any): any;
        mayCreate(dir: any, name: any): any;
        mayDelete(dir: any, name: any, isdir: any): any;
        mayOpen(node: any, flags: any): any;
        checkOpExists(op: any, err: any): any;
        MAX_OPEN_FDS: number;
        nextfd(): number;
        getStreamChecked(fd: any): any;
        getStream: (fd: any) => any;
        createStream(stream: any, fd?: number): any;
        closeStream(fd: any): void;
        dupStream(origStream: any, fd?: number): any;
        doSetAttr(stream: any, node: any, attr: any): void;
        chrdev_stream_ops: {
            open(stream: any): void;
            llseek(): never;
        };
        major: (dev: any) => number;
        minor: (dev: any) => number;
        makedev: (ma: any, mi: any) => number;
        registerDevice(dev: any, ops: any): void;
        getDevice: (dev: any) => any;
        getMounts(mount: any): any[];
        syncfs(populate: any, callback: any): void;
        mount(type: any, opts: any, mountpoint: any): any;
        unmount(mountpoint: any): void;
        lookup(parent: any, name: any): any;
        mknod(path: any, mode: any, dev: any): any;
        statfs(path: any): any;
        statfsStream(stream: any): any;
        statfsNode(node: any): {
            bsize: number;
            frsize: number;
            blocks: number;
            bfree: number;
            bavail: number;
            files: any;
            ffree: number;
            fsid: number;
            flags: number;
            namelen: number;
        };
        create(path: any, mode?: number): any;
        mkdir(path: any, mode?: number): any;
        mkdirTree(path: any, mode: any): void;
        mkdev(path: any, mode: any, dev: any): any;
        symlink(oldpath: any, newpath: any): any;
        link(oldpath: any, newpath: any, flags: any): any;
        rename(old_path: any, new_path: any): void;
        rmdir(path: any): void;
        readdir(path: any): any;
        unlink(path: any): void;
        readlink(path: any): any;
        stat(path: any, dontFollow: any): any;
        fstat(fd: any): any;
        lstat(path: any): any;
        doChmod(stream: any, node: any, mode: any, dontFollow: any): void;
        chmod(path: any, mode: any, dontFollow: any): void;
        lchmod(path: any, mode: any): void;
        fchmod(fd: any, mode: any): void;
        doChown(stream: any, node: any, dontFollow: any): void;
        chown(path: any, uid: any, gid: any, dontFollow: any): void;
        lchown(path: any, uid: any, gid: any): void;
        fchown(fd: any, uid: any, gid: any): void;
        doTruncate(stream: any, node: any, len: any): void;
        truncate(path: any, len: any): void;
        ftruncate(fd: any, len: any): void;
        utime(path: any, atime: any, mtime: any, dontFollow: any): void;
        open(path: any, flags: any, mode?: number): any;
        close(stream: any): void;
        isClosed(stream: any): boolean;
        llseek(stream: any, offset: any, whence: any): any;
        read(stream: any, buffer: any, offset: any, length: any, position: any): any;
        write(stream: any, buffer: any, offset: any, length: any, position: any, canOwn: any): any;
        mmap(stream: any, length: any, position: any, prot: any, flags: any): any;
        msync(stream: any, buffer: any, offset: any, length: any, mmapFlags: any): any;
        ioctl(stream: any, cmd: any, arg: any): any;
        readFile(path: any, opts?: {}): Uint8Array<any>;
        writeFile(path: any, data: any, opts?: {}): void;
        cwd: () => any;
        chdir(path: any): void;
        createDefaultDirectories(): void;
        createDefaultDevices(): void;
        createSpecialDirectories(): void;
        createStandardStreams(input: any, output: any, error: any): void;
        staticInit(): void;
        init(input: any, output: any, error: any): void;
        quit(): void;
        findObject(path: any, dontResolveLastLink: any): any;
        analyzePath(path: any, dontResolveLastLink: any): {
            isRoot: boolean;
            exists: boolean;
            error: number;
            name: null;
            path: null;
            object: null;
            parentExists: boolean;
            parentPath: null;
            parentObject: null;
        };
        createPath(parent: any, path: any, canRead: any, canWrite: any): any;
        createFile(parent: any, name: any, properties: any, canRead: any, canWrite: any): any;
        createDataFile(parent: any, name: any, data: any, canRead: any, canWrite: any, canOwn: any): void;
        createDevice(parent: any, name: any, input: any, output: any): any;
        forceLoadFile(obj: any): true | undefined;
        createLazyFile(parent: any, name: any, url: any, canRead: any, canWrite: any): any;
    };
    WORKERFS: {
        DIR_MODE: number;
        FILE_MODE: number;
        reader: null;
        mount(mount: any): any;
        createNode(parent: any, name: any, mode: any, dev: any, contents: any, mtime: any): any;
        node_ops: {
            getattr(node: any): {
                dev: number;
                ino: any;
                mode: any;
                nlink: number;
                uid: number;
                gid: number;
                rdev: number;
                size: any;
                atime: Date;
                mtime: Date;
                ctime: Date;
                blksize: number;
                blocks: number;
            };
            setattr(node: any, attr: any): void;
            lookup(parent: any, name: any): never;
            mknod(parent: any, name: any, mode: any, dev: any): never;
            rename(oldNode: any, newDir: any, newName: any): never;
            unlink(parent: any, name: any): never;
            rmdir(parent: any, name: any): never;
            readdir(node: any): string[];
            symlink(parent: any, newName: any, oldPath: any): never;
        };
        stream_ops: {
            read(stream: any, buffer: any, offset: any, length: any, position: any): any;
            write(stream: any, buffer: any, offset: any, length: any, position: any): never;
            llseek(stream: any, offset: any, whence: any): any;
        };
    };
    /** @param {string=} sig */
    addFunction: (func: any, sig?: string | undefined) => any;
    removeFunction: (index: any) => void;
    /**
     * Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the
     * emscripten HEAP, returns a copy of that string as a Javascript String object.
     *
     * @param {number} ptr
     * @param {number=} maxBytesToRead - An optional length that specifies the
     *   maximum number of bytes to read. You can omit this parameter to scan the
     *   string until the first 0 byte. If maxBytesToRead is passed, and the string
     *   at [ptr, ptr+maxBytesToReadr[ contains a null byte in the middle, then the
     *   string will cut short at that byte index.
     * @param {boolean=} ignoreNul - If true, the function will not stop on a NUL character.
     * @return {string}
     */
    UTF8ToString: (ptr: number, maxBytesToRead?: number | undefined, ignoreNul?: boolean | undefined) => string;
    stringToUTF8: (str: any, outPtr: any, maxBytesToWrite: any) => any;
    lengthBytesUTF8: (str: any) => number;
    /**
     * @param {string|null=} returnType
     * @param {Array=} argTypes
     * @param {Array=} args
     * @param {Object=} opts
     */
    ccall: (ident: any, returnType?: (string | null) | undefined, argTypes?: any[] | undefined, args?: any[] | undefined, opts?: Object | undefined) => any;
    /**
     * @param {string=} returnType
     * @param {Array=} argTypes
     * @param {Object=} opts
     */
    cwrap: (ident: any, returnType?: string | undefined, argTypes?: any[] | undefined, opts?: Object | undefined) => (...args: any[]) => any;
    FS_createPath: (...args: any[]) => any;
    FS_createDataFile: (...args: any[]) => any;
    FS_preloadFile: (parent: any, name: any, url: any, canRead: any, canWrite: any, dontCreateFile: any, canOwn: any, preFinish: any) => Promise<void>;
    FS_unlink: (...args: any[]) => any;
    FS_createLazyFile: (...args: any[]) => any;
    FS_createDevice: (...args: any[]) => any;
    addRunDependency: (id: any) => void;
    removeRunDependency: (id: any) => void;
};
interface WasmModule {
  _lcb_malloc(_0: BigInt): BigInt;
  _malloc(_0: BigInt): BigInt;
  _lcb_free(_0: BigInt): void;
  _free(_0: BigInt): void;
  _lcb_pointer_bytes(): number;
  _lcb_abi_version(): number;
  _lcb_ggml_backend_alloc_buffer(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_alloc_buffer(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_alloc_ctx_tensors(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_alloc_ctx_tensors(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_alloc_ctx_tensors_from_buft(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_alloc_ctx_tensors_from_buft(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_alloc_ctx_tensors_from_buft_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_alloc_ctx_tensors_from_buft_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_clear(_0: BigInt, _1: number): Promise<void>;
  _ggml_backend_buffer_clear(_0: BigInt, _1: number): Promise<void>;
  _lcb_ggml_backend_buffer_free(_0: BigInt): Promise<void>;
  _ggml_backend_buffer_free(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_buffer_get_alignment(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_alignment(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_alloc_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_alloc_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_base(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_base(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_max_size(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_max_size(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_size(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_size(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_type(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_get_type(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_get_usage(_0: BigInt): Promise<number>;
  _ggml_backend_buffer_get_usage(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_buffer_init_tensor(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_buffer_init_tensor(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_buffer_is_host(_0: BigInt): Promise<number>;
  _ggml_backend_buffer_is_host(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_buffer_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buffer_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buffer_reset(_0: BigInt): Promise<void>;
  _ggml_backend_buffer_reset(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_buffer_set_usage(_0: BigInt, _1: number): Promise<void>;
  _ggml_backend_buffer_set_usage(_0: BigInt, _1: number): Promise<void>;
  _lcb_ggml_backend_buft_alloc_buffer(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_buft_alloc_buffer(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buft_get_alignment(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buft_get_alignment(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buft_get_alloc_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_buft_get_alloc_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buft_get_device(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buft_get_device(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buft_get_max_size(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buft_get_max_size(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_buft_is_host(_0: BigInt): Promise<number>;
  _ggml_backend_buft_is_host(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_buft_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_buft_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_compare_graph_backend(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<number>;
  _ggml_backend_compare_graph_backend(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<number>;
  _lcb_ggml_backend_cpu_buffer_from_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_cpu_buffer_from_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_cpu_buffer_type(): Promise<BigInt>;
  _ggml_backend_cpu_buffer_type(): Promise<BigInt>;
  _lcb_ggml_backend_cpu_init(): Promise<BigInt>;
  _ggml_backend_cpu_init(): Promise<BigInt>;
  _lcb_ggml_backend_cpu_reg(): Promise<BigInt>;
  _ggml_backend_cpu_reg(): Promise<BigInt>;
  _lcb_ggml_backend_cpu_set_abort_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_cpu_set_abort_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_cpu_set_n_threads(_0: BigInt, _1: number): Promise<void>;
  _ggml_backend_cpu_set_n_threads(_0: BigInt, _1: number): Promise<void>;
  _lcb_ggml_backend_cpu_set_threadpool(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_cpu_set_threadpool(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_cpu_set_use_ref(_0: BigInt, _1: number): Promise<void>;
  _ggml_backend_cpu_set_use_ref(_0: BigInt, _1: number): Promise<void>;
  _lcb_ggml_backend_dev_backend_reg(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_backend_reg(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_buffer_from_host_ptr(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _ggml_backend_dev_buffer_from_host_ptr(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_buffer_type(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_buffer_type(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_by_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_by_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_by_type(_0: number): Promise<BigInt>;
  _ggml_backend_dev_by_type(_0: number): Promise<BigInt>;
  _lcb_ggml_backend_dev_count(): Promise<BigInt>;
  _ggml_backend_dev_count(): Promise<BigInt>;
  _lcb_ggml_backend_dev_description(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_description(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_get(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_get(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_get_props(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_dev_get_props(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_dev_host_buffer_type(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_host_buffer_type(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_dev_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_memory(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_dev_memory(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_dev_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_dev_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_dev_offload_op(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_dev_offload_op(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_dev_supports_buft(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_dev_supports_buft(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_dev_supports_op(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_dev_supports_op(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_dev_type(_0: BigInt): Promise<number>;
  _ggml_backend_dev_type(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_device_register(_0: BigInt): Promise<void>;
  _ggml_backend_device_register(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_event_free(_0: BigInt): Promise<void>;
  _ggml_backend_event_free(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_event_new(_0: BigInt): Promise<BigInt>;
  _ggml_backend_event_new(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_event_record(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_event_record(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_event_synchronize(_0: BigInt): Promise<void>;
  _ggml_backend_event_synchronize(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_event_wait(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_event_wait(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_free(_0: BigInt): Promise<void>;
  _ggml_backend_free(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_get_alignment(_0: BigInt): Promise<BigInt>;
  _ggml_backend_get_alignment(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_get_default_buffer_type(_0: BigInt): Promise<BigInt>;
  _ggml_backend_get_default_buffer_type(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_get_device(_0: BigInt): Promise<BigInt>;
  _ggml_backend_get_device(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_get_max_size(_0: BigInt): Promise<BigInt>;
  _ggml_backend_get_max_size(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_graph_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_graph_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_graph_compute_async(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_graph_compute_async(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_graph_copy(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_graph_copy(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_graph_copy_free(_0: BigInt): Promise<void>;
  _ggml_backend_graph_copy_free(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_graph_plan_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_graph_plan_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_graph_plan_create(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_graph_plan_create(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_graph_plan_free(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_graph_plan_free(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_guid(_0: BigInt): Promise<BigInt>;
  _ggml_backend_guid(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_init_best(): Promise<BigInt>;
  _ggml_backend_init_best(): Promise<BigInt>;
  _lcb_ggml_backend_init_by_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_init_by_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_init_by_type(_0: number, _1: BigInt): Promise<BigInt>;
  _ggml_backend_init_by_type(_0: number, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_is_cpu(_0: BigInt): Promise<number>;
  _ggml_backend_is_cpu(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_load(_0: BigInt): Promise<BigInt>;
  _ggml_backend_load(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_load_all(): Promise<void>;
  _ggml_backend_load_all(): Promise<void>;
  _lcb_ggml_backend_load_all_from_path(_0: BigInt): Promise<void>;
  _ggml_backend_load_all_from_path(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_meta_device(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _ggml_backend_meta_device(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_meta_split_axis_name(_0: number): Promise<BigInt>;
  _ggml_backend_meta_split_axis_name(_0: number): Promise<BigInt>;
  _lcb_ggml_backend_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_offload_op(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_offload_op(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_reg_by_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_reg_by_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_reg_count(): Promise<BigInt>;
  _ggml_backend_reg_count(): Promise<BigInt>;
  _lcb_ggml_backend_reg_dev_count(_0: BigInt): Promise<BigInt>;
  _ggml_backend_reg_dev_count(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_reg_dev_get(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_reg_dev_get(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_reg_get(_0: BigInt): Promise<BigInt>;
  _ggml_backend_reg_get(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_reg_get_proc_address(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_reg_get_proc_address(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_reg_name(_0: BigInt): Promise<BigInt>;
  _ggml_backend_reg_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_register(_0: BigInt): Promise<void>;
  _ggml_backend_register(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_alloc_graph(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_sched_alloc_graph(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_free(_0: BigInt): Promise<void>;
  _ggml_backend_sched_free(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_get_backend(_0: BigInt, _1: number): Promise<BigInt>;
  _ggml_backend_sched_get_backend(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_ggml_backend_sched_get_buffer_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_sched_get_buffer_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_sched_get_buffer_type(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_sched_get_buffer_type(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_sched_get_n_backends(_0: BigInt): Promise<number>;
  _ggml_backend_sched_get_n_backends(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_get_n_copies(_0: BigInt): Promise<number>;
  _ggml_backend_sched_get_n_copies(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_get_n_splits(_0: BigInt): Promise<number>;
  _ggml_backend_sched_get_n_splits(_0: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_get_tensor_backend(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_backend_sched_get_tensor_backend(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_backend_sched_graph_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_sched_graph_compute(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_graph_compute_async(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_sched_graph_compute_async(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_new(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number): Promise<BigInt>;
  _ggml_backend_sched_new(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number): Promise<BigInt>;
  _lcb_ggml_backend_sched_reserve(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_sched_reserve(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_sched_reserve_size(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_sched_reserve_size(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_reset(_0: BigInt): Promise<void>;
  _ggml_backend_sched_reset(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_set_eval_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_sched_set_eval_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_set_tensor_backend(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _ggml_backend_sched_set_tensor_backend(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_split_graph(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_sched_split_graph(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_sched_synchronize(_0: BigInt): Promise<void>;
  _ggml_backend_sched_synchronize(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_supports_buft(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_supports_buft(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_supports_op(_0: BigInt, _1: BigInt): Promise<number>;
  _ggml_backend_supports_op(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_ggml_backend_synchronize(_0: BigInt): Promise<void>;
  _ggml_backend_synchronize(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_alloc(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _ggml_backend_tensor_alloc(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _lcb_ggml_backend_tensor_copy(_0: BigInt, _1: BigInt): Promise<void>;
  _ggml_backend_tensor_copy(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_copy_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _ggml_backend_tensor_copy_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_get(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _ggml_backend_tensor_get(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_get_2d(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _ggml_backend_tensor_get_2d(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_get_2d_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt, _7: BigInt): Promise<void>;
  _ggml_backend_tensor_get_2d_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt, _7: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_get_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<void>;
  _ggml_backend_tensor_get_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_memset(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<void>;
  _ggml_backend_tensor_memset(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_set(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _ggml_backend_tensor_set(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_set_2d(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _ggml_backend_tensor_set_2d(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_set_2d_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt, _7: BigInt): Promise<void>;
  _ggml_backend_tensor_set_2d_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt, _7: BigInt): Promise<void>;
  _lcb_ggml_backend_tensor_set_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<void>;
  _ggml_backend_tensor_set_async(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<void>;
  _lcb_ggml_backend_unload(_0: BigInt): Promise<void>;
  _ggml_backend_unload(_0: BigInt): Promise<void>;
  _lcb_ggml_backend_view_init(_0: BigInt): Promise<number>;
  _ggml_backend_view_init(_0: BigInt): Promise<number>;
  _lcb_ggml_blck_size(_0: number): Promise<BigInt>;
  _ggml_blck_size(_0: number): Promise<BigInt>;
  _lcb_ggml_get_name(_0: BigInt): Promise<BigInt>;
  _ggml_get_name(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_is_quantized(_0: number): Promise<number>;
  _ggml_is_quantized(_0: number): Promise<number>;
  _lcb_ggml_nbytes(_0: BigInt): Promise<BigInt>;
  _ggml_nbytes(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_nelements(_0: BigInt): Promise<BigInt>;
  _ggml_nelements(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_nrows(_0: BigInt): Promise<BigInt>;
  _ggml_nrows(_0: BigInt): Promise<BigInt>;
  _lcb_ggml_row_size(_0: number, _1: BigInt): Promise<BigInt>;
  _ggml_row_size(_0: number, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_set_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _ggml_set_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_ggml_type_name(_0: number): Promise<BigInt>;
  _ggml_type_name(_0: number): Promise<BigInt>;
  _lcb_ggml_type_size(_0: number): Promise<BigInt>;
  _ggml_type_size(_0: number): Promise<BigInt>;
  _lcb_ggml_validate_row_data(_0: number, _1: BigInt, _2: BigInt): Promise<number>;
  _ggml_validate_row_data(_0: number, _1: BigInt, _2: BigInt): Promise<number>;
  _lcb_gguf_add_tensor(_0: BigInt, _1: BigInt): Promise<void>;
  _gguf_add_tensor(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_gguf_find_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_find_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_find_tensor(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_find_tensor(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_free(_0: BigInt): Promise<void>;
  _gguf_free(_0: BigInt): Promise<void>;
  _lcb_gguf_get_alignment(_0: BigInt): Promise<BigInt>;
  _gguf_get_alignment(_0: BigInt): Promise<BigInt>;
  _lcb_gguf_get_arr_data(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_arr_data(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_arr_n(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_arr_n(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_arr_str(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _gguf_get_arr_str(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_gguf_get_arr_type(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_arr_type(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_data_offset(_0: BigInt): Promise<BigInt>;
  _gguf_get_data_offset(_0: BigInt): Promise<BigInt>;
  _lcb_gguf_get_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_kv_type(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_kv_type(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_meta_data(_0: BigInt, _1: BigInt): Promise<void>;
  _gguf_get_meta_data(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_gguf_get_meta_size(_0: BigInt): Promise<BigInt>;
  _gguf_get_meta_size(_0: BigInt): Promise<BigInt>;
  _lcb_gguf_get_n_kv(_0: BigInt): Promise<BigInt>;
  _gguf_get_n_kv(_0: BigInt): Promise<BigInt>;
  _lcb_gguf_get_n_tensors(_0: BigInt): Promise<BigInt>;
  _gguf_get_n_tensors(_0: BigInt): Promise<BigInt>;
  _lcb_gguf_get_tensor_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_tensor_name(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_tensor_ne(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_tensor_ne(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_tensor_offset(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_tensor_offset(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_tensor_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_tensor_size(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_tensor_type(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_tensor_type(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_bool(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_bool(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_data(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_val_data(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_val_f32(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_f32(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_f64(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_f64(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_i16(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_i16(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_i32(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_i32(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_i64(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_val_i64(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_val_i8(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_i8(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_str(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_val_str(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_val_u16(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_u16(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_u32(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_u32(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_val_u64(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_get_val_u64(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_get_val_u8(_0: BigInt, _1: BigInt): Promise<number>;
  _gguf_get_val_u8(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_gguf_get_version(_0: BigInt): Promise<number>;
  _gguf_get_version(_0: BigInt): Promise<number>;
  _lcb_gguf_init_empty(): Promise<BigInt>;
  _gguf_init_empty(): Promise<BigInt>;
  _lcb_gguf_init_from_buffer(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _gguf_init_from_buffer(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_gguf_init_from_callback(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<BigInt>;
  _gguf_init_from_callback(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<BigInt>;
  _lcb_gguf_init_from_file(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_init_from_file(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_init_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_init_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_remove_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _gguf_remove_key(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_gguf_set_arr_data(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt): Promise<void>;
  _gguf_set_arr_data(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt): Promise<void>;
  _lcb_gguf_set_arr_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _gguf_set_arr_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<void>;
  _lcb_gguf_set_kv(_0: BigInt, _1: BigInt): Promise<void>;
  _gguf_set_kv(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_gguf_set_tensor_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _gguf_set_tensor_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_gguf_set_tensor_type(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_tensor_type(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_bool(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_bool(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_f32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_f32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_f64(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_f64(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_i16(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_i16(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_i32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_i32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_i64(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _gguf_set_val_i64(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_gguf_set_val_i8(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_i8(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_str(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _gguf_set_val_str(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_gguf_set_val_u16(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_u16(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_u32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_u32(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_set_val_u64(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _gguf_set_val_u64(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_gguf_set_val_u8(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _gguf_set_val_u8(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_gguf_type_name(_0: number): Promise<BigInt>;
  _gguf_type_name(_0: number): Promise<BigInt>;
  _lcb_gguf_write_to_file(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _gguf_write_to_file(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _lcb_gguf_write_to_file_ptr(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _gguf_write_to_file_ptr(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _lcb_llama_adapter_get_alora_invocation_tokens(_0: BigInt): Promise<BigInt>;
  _llama_adapter_get_alora_invocation_tokens(_0: BigInt): Promise<BigInt>;
  _lcb_llama_adapter_get_alora_n_invocation_tokens(_0: BigInt): Promise<BigInt>;
  _llama_adapter_get_alora_n_invocation_tokens(_0: BigInt): Promise<BigInt>;
  _lcb_llama_adapter_lora_free(_0: BigInt): Promise<void>;
  _llama_adapter_lora_free(_0: BigInt): Promise<void>;
  _lcb_llama_adapter_lora_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_adapter_lora_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_adapter_lora_init_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_adapter_lora_init_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_adapter_meta_count(_0: BigInt): Promise<number>;
  _llama_adapter_meta_count(_0: BigInt): Promise<number>;
  _lcb_llama_adapter_meta_key_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_adapter_meta_key_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_adapter_meta_val_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_adapter_meta_val_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_adapter_meta_val_str_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_adapter_meta_val_str_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_attach_threadpool(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _llama_attach_threadpool(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_llama_backend_free(): Promise<void>;
  _llama_backend_free(): Promise<void>;
  _lcb_llama_backend_init(): Promise<void>;
  _llama_backend_init(): Promise<void>;
  _lcb_llama_batch_free(_0: BigInt): Promise<void>;
  _llama_batch_free(_0: BigInt): Promise<void>;
  _lcb_llama_batch_get_one(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _llama_batch_get_one(_0: BigInt, _1: BigInt, _2: number): Promise<void>;
  _lcb_llama_batch_init(_0: BigInt, _1: number, _2: number, _3: number): Promise<void>;
  _llama_batch_init(_0: BigInt, _1: number, _2: number, _3: number): Promise<void>;
  _lcb_llama_chat_apply_template(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: BigInt, _5: number): Promise<number>;
  _llama_chat_apply_template(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: BigInt, _5: number): Promise<number>;
  _lcb_llama_chat_builtin_templates(_0: BigInt, _1: BigInt): Promise<number>;
  _llama_chat_builtin_templates(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_llama_context_default_params(_0: BigInt): Promise<void>;
  _llama_context_default_params(_0: BigInt): Promise<void>;
  _lcb_llama_decode(_0: BigInt, _1: BigInt): Promise<number>;
  _llama_decode(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_llama_detach_threadpool(_0: BigInt): Promise<void>;
  _llama_detach_threadpool(_0: BigInt): Promise<void>;
  _lcb_llama_detokenize(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number, _6: number): Promise<number>;
  _llama_detokenize(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number, _6: number): Promise<number>;
  _lcb_llama_encode(_0: BigInt, _1: BigInt): Promise<number>;
  _llama_encode(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_llama_flash_attn_type_name(_0: number): Promise<BigInt>;
  _llama_flash_attn_type_name(_0: number): Promise<BigInt>;
  _lcb_llama_free(_0: BigInt): Promise<void>;
  _llama_free(_0: BigInt): Promise<void>;
  _lcb_llama_ftype_name(_0: number): Promise<BigInt>;
  _llama_ftype_name(_0: number): Promise<BigInt>;
  _lcb_llama_get_embeddings(_0: BigInt): Promise<BigInt>;
  _llama_get_embeddings(_0: BigInt): Promise<BigInt>;
  _lcb_llama_get_embeddings_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_embeddings_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_embeddings_seq(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_embeddings_seq(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_logits(_0: BigInt): Promise<BigInt>;
  _llama_get_logits(_0: BigInt): Promise<BigInt>;
  _lcb_llama_get_logits_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_logits_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_memory(_0: BigInt): Promise<BigInt>;
  _llama_get_memory(_0: BigInt): Promise<BigInt>;
  _lcb_llama_get_model(_0: BigInt): Promise<BigInt>;
  _llama_get_model(_0: BigInt): Promise<BigInt>;
  _lcb_llama_get_sampled_candidates_count_ith(_0: BigInt, _1: number): Promise<number>;
  _llama_get_sampled_candidates_count_ith(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_get_sampled_candidates_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_sampled_candidates_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_sampled_logits_count_ith(_0: BigInt, _1: number): Promise<number>;
  _llama_get_sampled_logits_count_ith(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_get_sampled_logits_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_sampled_logits_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_sampled_probs_count_ith(_0: BigInt, _1: number): Promise<number>;
  _llama_get_sampled_probs_count_ith(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_get_sampled_probs_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_get_sampled_probs_ith(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_get_sampled_token_ith(_0: BigInt, _1: number): Promise<number>;
  _llama_get_sampled_token_ith(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_init_from_model(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_init_from_model(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_load_mode_from_str(_0: BigInt): Promise<number>;
  _llama_load_mode_from_str(_0: BigInt): Promise<number>;
  _lcb_llama_load_mode_name(_0: number): Promise<BigInt>;
  _llama_load_mode_name(_0: number): Promise<BigInt>;
  _lcb_llama_log_get(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_log_get(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_max_devices(): Promise<BigInt>;
  _llama_max_devices(): Promise<BigInt>;
  _lcb_llama_max_parallel_sequences(): Promise<BigInt>;
  _llama_max_parallel_sequences(): Promise<BigInt>;
  _lcb_llama_max_tensor_buft_overrides(): Promise<BigInt>;
  _llama_max_tensor_buft_overrides(): Promise<BigInt>;
  _lcb_llama_memory_can_shift(_0: BigInt): Promise<number>;
  _llama_memory_can_shift(_0: BigInt): Promise<number>;
  _lcb_llama_memory_clear(_0: BigInt, _1: number): Promise<void>;
  _llama_memory_clear(_0: BigInt, _1: number): Promise<void>;
  _lcb_llama_memory_seq_add(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _llama_memory_seq_add(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _lcb_llama_memory_seq_cp(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _llama_memory_seq_cp(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _lcb_llama_memory_seq_div(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _llama_memory_seq_div(_0: BigInt, _1: number, _2: number, _3: number, _4: number): Promise<void>;
  _lcb_llama_memory_seq_keep(_0: BigInt, _1: number): Promise<void>;
  _llama_memory_seq_keep(_0: BigInt, _1: number): Promise<void>;
  _lcb_llama_memory_seq_pos_max(_0: BigInt, _1: number): Promise<number>;
  _llama_memory_seq_pos_max(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_memory_seq_pos_min(_0: BigInt, _1: number): Promise<number>;
  _llama_memory_seq_pos_min(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_memory_seq_rm(_0: BigInt, _1: number, _2: number, _3: number): Promise<number>;
  _llama_memory_seq_rm(_0: BigInt, _1: number, _2: number, _3: number): Promise<number>;
  _lcb_llama_model_chat_template(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_model_chat_template(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_model_cls_label(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_model_cls_label(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_model_decoder_start_token(_0: BigInt): Promise<number>;
  _llama_model_decoder_start_token(_0: BigInt): Promise<number>;
  _lcb_llama_model_default_params(_0: BigInt): Promise<void>;
  _llama_model_default_params(_0: BigInt): Promise<void>;
  _lcb_llama_model_desc(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _llama_model_desc(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _lcb_llama_model_free(_0: BigInt): Promise<void>;
  _llama_model_free(_0: BigInt): Promise<void>;
  _lcb_llama_model_ftype(_0: BigInt): Promise<number>;
  _llama_model_ftype(_0: BigInt): Promise<number>;
  _lcb_llama_model_get_vocab(_0: BigInt): Promise<BigInt>;
  _llama_model_get_vocab(_0: BigInt): Promise<BigInt>;
  _lcb_llama_model_has_decoder(_0: BigInt): Promise<number>;
  _llama_model_has_decoder(_0: BigInt): Promise<number>;
  _lcb_llama_model_has_encoder(_0: BigInt): Promise<number>;
  _llama_model_has_encoder(_0: BigInt): Promise<number>;
  _lcb_llama_model_init_from_user(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _llama_model_init_from_user(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _lcb_llama_model_is_diffusion(_0: BigInt): Promise<number>;
  _llama_model_is_diffusion(_0: BigInt): Promise<number>;
  _lcb_llama_model_is_hybrid(_0: BigInt): Promise<number>;
  _llama_model_is_hybrid(_0: BigInt): Promise<number>;
  _lcb_llama_model_is_recurrent(_0: BigInt): Promise<number>;
  _llama_model_is_recurrent(_0: BigInt): Promise<number>;
  _lcb_llama_model_load_from_file(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_model_load_from_file(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_model_load_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_model_load_from_file_ptr(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_model_load_from_splits(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _llama_model_load_from_splits(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_llama_model_meta_count(_0: BigInt): Promise<number>;
  _llama_model_meta_count(_0: BigInt): Promise<number>;
  _lcb_llama_model_meta_key_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_model_meta_key_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_model_meta_key_str(_0: number): Promise<BigInt>;
  _llama_model_meta_key_str(_0: number): Promise<BigInt>;
  _lcb_llama_model_meta_val_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_model_meta_val_str(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_model_meta_val_str_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_model_meta_val_str_by_index(_0: BigInt, _1: number, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_model_n_cls_out(_0: BigInt): Promise<number>;
  _llama_model_n_cls_out(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_ctx_train(_0: BigInt): Promise<number>;
  _llama_model_n_ctx_train(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_embd(_0: BigInt): Promise<number>;
  _llama_model_n_embd(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_embd_inp(_0: BigInt): Promise<number>;
  _llama_model_n_embd_inp(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_embd_out(_0: BigInt): Promise<number>;
  _llama_model_n_embd_out(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_head(_0: BigInt): Promise<number>;
  _llama_model_n_head(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_head_kv(_0: BigInt): Promise<number>;
  _llama_model_n_head_kv(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_layer(_0: BigInt): Promise<number>;
  _llama_model_n_layer(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_layer_nextn(_0: BigInt): Promise<number>;
  _llama_model_n_layer_nextn(_0: BigInt): Promise<number>;
  _lcb_llama_model_n_params(_0: BigInt): Promise<BigInt>;
  _llama_model_n_params(_0: BigInt): Promise<BigInt>;
  _lcb_llama_model_n_swa(_0: BigInt): Promise<number>;
  _llama_model_n_swa(_0: BigInt): Promise<number>;
  _lcb_llama_model_quantize(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _llama_model_quantize(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _lcb_llama_model_quantize_default_params(_0: BigInt): Promise<void>;
  _llama_model_quantize_default_params(_0: BigInt): Promise<void>;
  _lcb_llama_model_rope_freq_scale_train(_0: BigInt): Promise<number>;
  _llama_model_rope_freq_scale_train(_0: BigInt): Promise<number>;
  _lcb_llama_model_rope_type(_0: BigInt): Promise<number>;
  _llama_model_rope_type(_0: BigInt): Promise<number>;
  _lcb_llama_model_save_to_file(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_model_save_to_file(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_model_size(_0: BigInt): Promise<BigInt>;
  _llama_model_size(_0: BigInt): Promise<BigInt>;
  _lcb_llama_n_batch(_0: BigInt): Promise<number>;
  _llama_n_batch(_0: BigInt): Promise<number>;
  _lcb_llama_n_ctx(_0: BigInt): Promise<number>;
  _llama_n_ctx(_0: BigInt): Promise<number>;
  _lcb_llama_n_ctx_seq(_0: BigInt): Promise<number>;
  _llama_n_ctx_seq(_0: BigInt): Promise<number>;
  _lcb_llama_n_rs_seq(_0: BigInt): Promise<number>;
  _llama_n_rs_seq(_0: BigInt): Promise<number>;
  _lcb_llama_n_seq_max(_0: BigInt): Promise<number>;
  _llama_n_seq_max(_0: BigInt): Promise<number>;
  _lcb_llama_n_threads(_0: BigInt): Promise<number>;
  _llama_n_threads(_0: BigInt): Promise<number>;
  _lcb_llama_n_threads_batch(_0: BigInt): Promise<number>;
  _llama_n_threads_batch(_0: BigInt): Promise<number>;
  _lcb_llama_n_ubatch(_0: BigInt): Promise<number>;
  _llama_n_ubatch(_0: BigInt): Promise<number>;
  _lcb_llama_numa_init(_0: number): Promise<void>;
  _llama_numa_init(_0: number): Promise<void>;
  _lcb_llama_opt_epoch(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _llama_opt_epoch(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<void>;
  _lcb_llama_opt_init(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _llama_opt_init(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_llama_opt_param_filter_all(_0: BigInt, _1: BigInt): Promise<number>;
  _llama_opt_param_filter_all(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_llama_perf_context(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_perf_context(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_perf_context_print(_0: BigInt): Promise<void>;
  _llama_perf_context_print(_0: BigInt): Promise<void>;
  _lcb_llama_perf_context_reset(_0: BigInt): Promise<void>;
  _llama_perf_context_reset(_0: BigInt): Promise<void>;
  _lcb_llama_perf_sampler(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_perf_sampler(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_perf_sampler_print(_0: BigInt): Promise<void>;
  _llama_perf_sampler_print(_0: BigInt): Promise<void>;
  _lcb_llama_perf_sampler_reset(_0: BigInt): Promise<void>;
  _llama_perf_sampler_reset(_0: BigInt): Promise<void>;
  _lcb_llama_pooling_type(_0: BigInt): Promise<number>;
  _llama_pooling_type(_0: BigInt): Promise<number>;
  _lcb_llama_print_system_info(): Promise<BigInt>;
  _llama_print_system_info(): Promise<BigInt>;
  _lcb_llama_sampler_accept(_0: BigInt, _1: number): Promise<void>;
  _llama_sampler_accept(_0: BigInt, _1: number): Promise<void>;
  _lcb_llama_sampler_apply(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_sampler_apply(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_sampler_chain_add(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_sampler_chain_add(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_sampler_chain_default_params(_0: BigInt): Promise<void>;
  _llama_sampler_chain_default_params(): Promise<number>;
  _lcb_llama_sampler_chain_get(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_sampler_chain_get(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_sampler_chain_init(_0: BigInt): Promise<BigInt>;
  _llama_sampler_chain_init(_0: number): Promise<BigInt>;
  _lcb_llama_sampler_chain_n(_0: BigInt): Promise<number>;
  _llama_sampler_chain_n(_0: BigInt): Promise<number>;
  _lcb_llama_sampler_chain_remove(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_sampler_chain_remove(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_sampler_clone(_0: BigInt): Promise<BigInt>;
  _llama_sampler_clone(_0: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_copy(_0: BigInt, _1: BigInt): Promise<void>;
  _llama_sampler_copy(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_llama_sampler_free(_0: BigInt): Promise<void>;
  _llama_sampler_free(_0: BigInt): Promise<void>;
  _lcb_llama_sampler_get_seed(_0: BigInt): Promise<number>;
  _llama_sampler_get_seed(_0: BigInt): Promise<number>;
  _lcb_llama_sampler_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_sampler_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_adaptive_p(_0: number, _1: number, _2: number): Promise<BigInt>;
  _llama_sampler_init_adaptive_p(_0: number, _1: number, _2: number): Promise<BigInt>;
  _lcb_llama_sampler_init_dist(_0: number): Promise<BigInt>;
  _llama_sampler_init_dist(_0: number): Promise<BigInt>;
  _lcb_llama_sampler_init_dry(_0: BigInt, _1: number, _2: number, _3: number, _4: number, _5: BigInt, _6: BigInt): Promise<BigInt>;
  _llama_sampler_init_dry(_0: BigInt, _1: number, _2: number, _3: number, _4: number, _5: BigInt, _6: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_grammar(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _llama_sampler_init_grammar(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_grammar_lazy_patterns(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<BigInt>;
  _llama_sampler_init_grammar_lazy_patterns(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt, _5: BigInt, _6: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_greedy(): Promise<BigInt>;
  _llama_sampler_init_greedy(): Promise<BigInt>;
  _lcb_llama_sampler_init_infill(_0: BigInt): Promise<BigInt>;
  _llama_sampler_init_infill(_0: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_logit_bias(_0: number, _1: number, _2: BigInt): Promise<BigInt>;
  _llama_sampler_init_logit_bias(_0: number, _1: number, _2: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_min_p(_0: number, _1: BigInt): Promise<BigInt>;
  _llama_sampler_init_min_p(_0: number, _1: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_mirostat(_0: number, _1: number, _2: number, _3: number, _4: number): Promise<BigInt>;
  _llama_sampler_init_mirostat(_0: number, _1: number, _2: number, _3: number, _4: number): Promise<BigInt>;
  _lcb_llama_sampler_init_mirostat_v2(_0: number, _1: number, _2: number): Promise<BigInt>;
  _llama_sampler_init_mirostat_v2(_0: number, _1: number, _2: number): Promise<BigInt>;
  _lcb_llama_sampler_init_penalties(_0: number, _1: number, _2: number, _3: number, _4: number): Promise<BigInt>;
  _llama_sampler_init_penalties(_0: number, _1: number, _2: number, _3: number, _4: number): Promise<BigInt>;
  _lcb_llama_sampler_init_temp(_0: number): Promise<BigInt>;
  _llama_sampler_init_temp(_0: number): Promise<BigInt>;
  _lcb_llama_sampler_init_temp_ext(_0: number, _1: number, _2: number): Promise<BigInt>;
  _llama_sampler_init_temp_ext(_0: number, _1: number, _2: number): Promise<BigInt>;
  _lcb_llama_sampler_init_top_k(_0: number): Promise<BigInt>;
  _llama_sampler_init_top_k(_0: number): Promise<BigInt>;
  _lcb_llama_sampler_init_top_n_sigma(_0: number): Promise<BigInt>;
  _llama_sampler_init_top_n_sigma(_0: number): Promise<BigInt>;
  _lcb_llama_sampler_init_top_p(_0: number, _1: BigInt): Promise<BigInt>;
  _llama_sampler_init_top_p(_0: number, _1: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_typical(_0: number, _1: BigInt): Promise<BigInt>;
  _llama_sampler_init_typical(_0: number, _1: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_init_xtc(_0: number, _1: number, _2: BigInt, _3: number): Promise<BigInt>;
  _llama_sampler_init_xtc(_0: number, _1: number, _2: BigInt, _3: number): Promise<BigInt>;
  _lcb_llama_sampler_name(_0: BigInt): Promise<BigInt>;
  _llama_sampler_name(_0: BigInt): Promise<BigInt>;
  _lcb_llama_sampler_reset(_0: BigInt): Promise<void>;
  _llama_sampler_reset(_0: BigInt): Promise<void>;
  _lcb_llama_sampler_sample(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _llama_sampler_sample(_0: BigInt, _1: BigInt, _2: number): Promise<number>;
  _lcb_llama_set_abort_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _llama_set_abort_callback(_0: BigInt, _1: BigInt, _2: BigInt): Promise<void>;
  _lcb_llama_set_adapter_cvec(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number): Promise<number>;
  _llama_set_adapter_cvec(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number): Promise<number>;
  _lcb_llama_set_adapters_lora(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_set_adapters_lora(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_set_causal_attn(_0: BigInt, _1: number): Promise<void>;
  _llama_set_causal_attn(_0: BigInt, _1: number): Promise<void>;
  _lcb_llama_set_embeddings(_0: BigInt, _1: number): Promise<void>;
  _llama_set_embeddings(_0: BigInt, _1: number): Promise<void>;
  _lcb_llama_set_n_threads(_0: BigInt, _1: number, _2: number): Promise<void>;
  _llama_set_n_threads(_0: BigInt, _1: number, _2: number): Promise<void>;
  _lcb_llama_set_sampler(_0: BigInt, _1: number, _2: BigInt): Promise<number>;
  _llama_set_sampler(_0: BigInt, _1: number, _2: BigInt): Promise<number>;
  _lcb_llama_split_path(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<number>;
  _llama_split_path(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<number>;
  _lcb_llama_split_prefix(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<number>;
  _llama_split_prefix(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<number>;
  _lcb_llama_state_get_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _llama_state_get_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_llama_state_get_size(_0: BigInt): Promise<BigInt>;
  _llama_state_get_size(_0: BigInt): Promise<BigInt>;
  _lcb_llama_state_load_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _llama_state_load_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _lcb_llama_state_save_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _llama_state_save_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_llama_state_seq_get_data(_0: BigInt, _1: BigInt, _2: BigInt, _3: number): Promise<BigInt>;
  _llama_state_seq_get_data(_0: BigInt, _1: BigInt, _2: BigInt, _3: number): Promise<BigInt>;
  _lcb_llama_state_seq_get_data_ext(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<BigInt>;
  _llama_state_seq_get_data_ext(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<BigInt>;
  _lcb_llama_state_seq_get_size(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_state_seq_get_size(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_state_seq_get_size_ext(_0: BigInt, _1: number, _2: number): Promise<BigInt>;
  _llama_state_seq_get_size_ext(_0: BigInt, _1: number, _2: number): Promise<BigInt>;
  _lcb_llama_state_seq_load_file(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt, _5: BigInt): Promise<BigInt>;
  _llama_state_seq_load_file(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt, _5: BigInt): Promise<BigInt>;
  _lcb_llama_state_seq_save_file(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt): Promise<BigInt>;
  _llama_state_seq_save_file(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: BigInt): Promise<BigInt>;
  _lcb_llama_state_seq_set_data(_0: BigInt, _1: BigInt, _2: BigInt, _3: number): Promise<BigInt>;
  _llama_state_seq_set_data(_0: BigInt, _1: BigInt, _2: BigInt, _3: number): Promise<BigInt>;
  _lcb_llama_state_seq_set_data_ext(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<BigInt>;
  _llama_state_seq_set_data_ext(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number): Promise<BigInt>;
  _lcb_llama_state_set_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _llama_state_set_data(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_llama_supports_gpu_offload(): Promise<number>;
  _llama_supports_gpu_offload(): Promise<number>;
  _lcb_llama_supports_mlock(): Promise<number>;
  _llama_supports_mlock(): Promise<number>;
  _lcb_llama_supports_mmap(): Promise<number>;
  _llama_supports_mmap(): Promise<number>;
  _lcb_llama_supports_rpc(): Promise<number>;
  _llama_supports_rpc(): Promise<number>;
  _lcb_llama_synchronize(_0: BigInt): Promise<void>;
  _llama_synchronize(_0: BigInt): Promise<void>;
  _lcb_llama_time_us(): Promise<BigInt>;
  _llama_time_us(): Promise<BigInt>;
  _lcb_llama_token_to_piece(_0: BigInt, _1: number, _2: BigInt, _3: number, _4: number, _5: number): Promise<number>;
  _llama_token_to_piece(_0: BigInt, _1: number, _2: BigInt, _3: number, _4: number, _5: number): Promise<number>;
  _lcb_llama_tokenize(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number, _6: number): Promise<number>;
  _llama_tokenize(_0: BigInt, _1: BigInt, _2: number, _3: BigInt, _4: number, _5: number, _6: number): Promise<number>;
  _lcb_llama_version(): Promise<BigInt>;
  _llama_version(): Promise<BigInt>;
  _lcb_llama_vocab_bos(_0: BigInt): Promise<number>;
  _llama_vocab_bos(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_eos(_0: BigInt): Promise<number>;
  _llama_vocab_eos(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_eot(_0: BigInt): Promise<number>;
  _llama_vocab_eot(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_mid(_0: BigInt): Promise<number>;
  _llama_vocab_fim_mid(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_pad(_0: BigInt): Promise<number>;
  _llama_vocab_fim_pad(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_pre(_0: BigInt): Promise<number>;
  _llama_vocab_fim_pre(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_rep(_0: BigInt): Promise<number>;
  _llama_vocab_fim_rep(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_sep(_0: BigInt): Promise<number>;
  _llama_vocab_fim_sep(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_fim_suf(_0: BigInt): Promise<number>;
  _llama_vocab_fim_suf(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_get_add_bos(_0: BigInt): Promise<number>;
  _llama_vocab_get_add_bos(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_get_add_eos(_0: BigInt): Promise<number>;
  _llama_vocab_get_add_eos(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_get_add_sep(_0: BigInt): Promise<number>;
  _llama_vocab_get_add_sep(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_get_attr(_0: BigInt, _1: number): Promise<number>;
  _llama_vocab_get_attr(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_vocab_get_score(_0: BigInt, _1: number): Promise<number>;
  _llama_vocab_get_score(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_vocab_get_suppress_tokens(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _llama_vocab_get_suppress_tokens(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_llama_vocab_get_text(_0: BigInt, _1: number): Promise<BigInt>;
  _llama_vocab_get_text(_0: BigInt, _1: number): Promise<BigInt>;
  _lcb_llama_vocab_is_control(_0: BigInt, _1: number): Promise<number>;
  _llama_vocab_is_control(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_vocab_is_eog(_0: BigInt, _1: number): Promise<number>;
  _llama_vocab_is_eog(_0: BigInt, _1: number): Promise<number>;
  _lcb_llama_vocab_mask(_0: BigInt): Promise<number>;
  _llama_vocab_mask(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_n_tokens(_0: BigInt): Promise<number>;
  _llama_vocab_n_tokens(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_nl(_0: BigInt): Promise<number>;
  _llama_vocab_nl(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_pad(_0: BigInt): Promise<number>;
  _llama_vocab_pad(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_sep(_0: BigInt): Promise<number>;
  _llama_vocab_sep(_0: BigInt): Promise<number>;
  _lcb_llama_vocab_type(_0: BigInt): Promise<number>;
  _llama_vocab_type(_0: BigInt): Promise<number>;
  _lcb_mtmd_batch_add_chunk(_0: BigInt, _1: BigInt): Promise<number>;
  _mtmd_batch_add_chunk(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_mtmd_batch_encode(_0: BigInt): Promise<number>;
  _mtmd_batch_encode(_0: BigInt): Promise<number>;
  _lcb_mtmd_batch_free(_0: BigInt): Promise<void>;
  _mtmd_batch_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_batch_get_output_embd(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_batch_get_output_embd(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_batch_init(_0: BigInt): Promise<BigInt>;
  _mtmd_batch_init(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_free(_0: BigInt): Promise<void>;
  _mtmd_bitmap_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_bitmap_get_data(_0: BigInt): Promise<BigInt>;
  _mtmd_bitmap_get_data(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_get_id(_0: BigInt): Promise<BigInt>;
  _mtmd_bitmap_get_id(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_get_n_bytes(_0: BigInt): Promise<BigInt>;
  _mtmd_bitmap_get_n_bytes(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_get_nx(_0: BigInt): Promise<number>;
  _mtmd_bitmap_get_nx(_0: BigInt): Promise<number>;
  _lcb_mtmd_bitmap_get_ny(_0: BigInt): Promise<number>;
  _mtmd_bitmap_get_ny(_0: BigInt): Promise<number>;
  _lcb_mtmd_bitmap_init(_0: number, _1: number, _2: BigInt): Promise<BigInt>;
  _mtmd_bitmap_init(_0: number, _1: number, _2: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_init_from_audio(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_bitmap_init_from_audio(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_init_lazy(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _mtmd_bitmap_init_lazy(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<BigInt>;
  _lcb_mtmd_bitmap_is_audio(_0: BigInt): Promise<number>;
  _mtmd_bitmap_is_audio(_0: BigInt): Promise<number>;
  _lcb_mtmd_bitmap_set_id(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_bitmap_set_id(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_bitmap_set_mergeable(_0: BigInt, _1: number): Promise<void>;
  _mtmd_bitmap_set_mergeable(_0: BigInt, _1: number): Promise<void>;
  _lcb_mtmd_context_params_default(_0: BigInt): Promise<void>;
  _mtmd_context_params_default(_0: BigInt): Promise<void>;
  _lcb_mtmd_decode_use_mrope(_0: BigInt): Promise<number>;
  _mtmd_decode_use_mrope(_0: BigInt): Promise<number>;
  _lcb_mtmd_decode_use_non_causal(_0: BigInt, _1: BigInt): Promise<number>;
  _mtmd_decode_use_non_causal(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_mtmd_default_marker(): Promise<BigInt>;
  _mtmd_default_marker(): Promise<BigInt>;
  _lcb_mtmd_encode_chunk(_0: BigInt, _1: BigInt): Promise<number>;
  _mtmd_encode_chunk(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_mtmd_free(_0: BigInt): Promise<void>;
  _mtmd_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_gen_audio_get_info(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_gen_audio_get_info(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_gen_audio_process(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _mtmd_gen_audio_process(_0: BigInt, _1: BigInt, _2: BigInt): Promise<number>;
  _lcb_mtmd_gen_inp_default(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_gen_inp_default(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_get_audio_sample_rate(_0: BigInt): Promise<number>;
  _mtmd_get_audio_sample_rate(_0: BigInt): Promise<number>;
  _lcb_mtmd_get_cap_from_file(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_get_cap_from_file(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_get_marker(_0: BigInt): Promise<BigInt>;
  _mtmd_get_marker(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_get_output_embd(_0: BigInt): Promise<BigInt>;
  _mtmd_get_output_embd(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_helper_bitmap_init_from_buf(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number, _5: BigInt): Promise<void>;
  _mtmd_helper_bitmap_init_from_buf(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number, _5: BigInt): Promise<void>;
  _lcb_mtmd_helper_bitmap_init_from_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: BigInt): Promise<void>;
  _mtmd_helper_bitmap_init_from_file(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: BigInt): Promise<void>;
  _lcb_mtmd_helper_decode_image_chunk(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number, _5: number, _6: number, _7: BigInt, _8: BigInt, _9: BigInt): Promise<number>;
  _mtmd_helper_decode_image_chunk(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number, _5: number, _6: number, _7: BigInt, _8: BigInt, _9: BigInt): Promise<number>;
  _lcb_mtmd_helper_eval_chunk_single(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number, _6: number, _7: BigInt): Promise<number>;
  _mtmd_helper_eval_chunk_single(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number, _6: number, _7: BigInt): Promise<number>;
  _lcb_mtmd_helper_eval_chunks(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number, _6: number, _7: BigInt): Promise<number>;
  _mtmd_helper_eval_chunks(_0: BigInt, _1: BigInt, _2: BigInt, _3: number, _4: number, _5: number, _6: number, _7: BigInt): Promise<number>;
  _lcb_mtmd_helper_gen_audio_free(_0: BigInt): Promise<void>;
  _mtmd_helper_gen_audio_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_helper_gen_audio_get_output(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _mtmd_helper_gen_audio_get_output(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _lcb_mtmd_helper_gen_audio_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_helper_gen_audio_init(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_helper_gen_audio_reset(_0: BigInt): Promise<void>;
  _mtmd_helper_gen_audio_reset(_0: BigInt): Promise<void>;
  _lcb_mtmd_helper_gen_audio_set_input(_0: BigInt, _1: BigInt): Promise<number>;
  _mtmd_helper_gen_audio_set_input(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_mtmd_helper_gen_audio_step_gen(_0: BigInt, _1: number, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _mtmd_helper_gen_audio_step_gen(_0: BigInt, _1: number, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _lcb_mtmd_helper_gen_audio_step_prompt(_0: BigInt, _1: number): Promise<number>;
  _mtmd_helper_gen_audio_step_prompt(_0: BigInt, _1: number): Promise<number>;
  _lcb_mtmd_helper_get_n_pos(_0: BigInt): Promise<number>;
  _mtmd_helper_get_n_pos(_0: BigInt): Promise<number>;
  _lcb_mtmd_helper_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _mtmd_helper_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_helper_image_get_decoder_pos(_0: BigInt, _1: number, _2: BigInt): Promise<void>;
  _mtmd_helper_image_get_decoder_pos(_0: BigInt, _1: number, _2: BigInt): Promise<void>;
  _lcb_mtmd_helper_init_opt_default(_0: BigInt): Promise<void>;
  _mtmd_helper_init_opt_default(_0: BigInt): Promise<void>;
  _lcb_mtmd_helper_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_helper_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_helper_model_can_chat(_0: BigInt, _1: BigInt): Promise<number>;
  _mtmd_helper_model_can_chat(_0: BigInt, _1: BigInt): Promise<number>;
  _lcb_mtmd_helper_support_video(_0: BigInt): Promise<number>;
  _mtmd_helper_support_video(_0: BigInt): Promise<number>;
  _lcb_mtmd_helper_video_init_params_default(_0: BigInt): Promise<void>;
  _mtmd_helper_video_init_params_default(_0: BigInt): Promise<void>;
  _lcb_mtmd_image_tokens_get_decoder_pos(_0: BigInt, _1: BigInt, _2: number, _3: BigInt): Promise<void>;
  _mtmd_image_tokens_get_decoder_pos(_0: BigInt, _1: BigInt, _2: number, _3: BigInt): Promise<void>;
  _lcb_mtmd_image_tokens_get_id(_0: BigInt): Promise<BigInt>;
  _mtmd_image_tokens_get_id(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_image_tokens_get_n_pos(_0: BigInt): Promise<number>;
  _mtmd_image_tokens_get_n_pos(_0: BigInt): Promise<number>;
  _lcb_mtmd_image_tokens_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _mtmd_image_tokens_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_init_from_file(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _mtmd_init_from_file(_0: BigInt, _1: BigInt, _2: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_copy(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_copy(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_free(_0: BigInt): Promise<void>;
  _mtmd_input_chunk_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_input_chunk_get_id(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_get_id(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_get_n_pos(_0: BigInt): Promise<number>;
  _mtmd_input_chunk_get_n_pos(_0: BigInt): Promise<number>;
  _lcb_mtmd_input_chunk_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_get_n_tokens(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_get_placeholder(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_get_placeholder(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_get_tokens_image(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_get_tokens_image(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_get_tokens_text(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_get_tokens_text(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_get_type(_0: BigInt): Promise<number>;
  _mtmd_input_chunk_get_type(_0: BigInt): Promise<number>;
  _lcb_mtmd_input_chunk_load(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_input_chunk_load(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunk_save(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _mtmd_input_chunk_save(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt): Promise<number>;
  _lcb_mtmd_input_chunks_free(_0: BigInt): Promise<void>;
  _mtmd_input_chunks_free(_0: BigInt): Promise<void>;
  _lcb_mtmd_input_chunks_get(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _mtmd_input_chunks_get(_0: BigInt, _1: BigInt): Promise<BigInt>;
  _lcb_mtmd_input_chunks_init(): Promise<BigInt>;
  _mtmd_input_chunks_init(): Promise<BigInt>;
  _lcb_mtmd_input_chunks_size(_0: BigInt): Promise<BigInt>;
  _mtmd_input_chunks_size(_0: BigInt): Promise<BigInt>;
  _lcb_mtmd_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _mtmd_log_set(_0: BigInt, _1: BigInt): Promise<void>;
  _lcb_mtmd_support_audio(_0: BigInt): Promise<number>;
  _mtmd_support_audio(_0: BigInt): Promise<number>;
  _lcb_mtmd_support_vision(_0: BigInt): Promise<number>;
  _mtmd_support_vision(_0: BigInt): Promise<number>;
  _lcb_mtmd_test_create_input_chunks(): Promise<BigInt>;
  _mtmd_test_create_input_chunks(): Promise<BigInt>;
  _lcb_mtmd_tokenize(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _mtmd_tokenize(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: BigInt): Promise<number>;
  _lcb_mtmd_tokenize_from_parts(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number): Promise<number>;
  _mtmd_tokenize_from_parts(_0: BigInt, _1: BigInt, _2: BigInt, _3: BigInt, _4: number): Promise<number>;
  _lcb_sizeof_record(_0: number): BigInt;
  _lcb_alignof_record(_0: number): BigInt;
  _lcb_offsetof_field(_0: number, _1: number): BigInt;
  _lcb_sizeof_field(_0: number, _1: number): BigInt;
  _lcb_constant(_0: number): BigInt;
  _lcb_schema_hash(): BigInt;
}

type EmbindString = ArrayBuffer|Uint8Array|Uint8ClampedArray|Int8Array|string;
export interface ClassHandle {
  isAliasOf(other: ClassHandle): boolean;
  delete(): void;
  deleteLater(): this;
  isDeleted(): boolean;
  // @ts-ignore - If targeting lower than ESNext, this symbol might not exist.
  [Symbol.dispose](): void;
  clone(): this;
}
export interface common_chat_roleValue<T extends number> {
  value: T;
}
export type common_chat_role = common_chat_roleValue<0>|common_chat_roleValue<1>|common_chat_roleValue<2>|common_chat_roleValue<3>|common_chat_roleValue<4>;

export interface common_chat_tool_choiceValue<T extends number> {
  value: T;
}
export type common_chat_tool_choice = common_chat_tool_choiceValue<0>|common_chat_tool_choiceValue<1>|common_chat_tool_choiceValue<2>;

export interface common_chat_formatValue<T extends number> {
  value: T;
}
export type common_chat_format = common_chat_formatValue<0>|common_chat_formatValue<1>|common_chat_formatValue<2>|common_chat_formatValue<3>|common_chat_formatValue<4>|common_chat_formatValue<5>;

export interface common_chat_continuationValue<T extends number> {
  value: T;
}
export type common_chat_continuation = common_chat_continuationValue<0>|common_chat_continuationValue<1>|common_chat_continuationValue<2>|common_chat_continuationValue<3>;

export interface common_reasoning_formatValue<T extends number> {
  value: T;
}
export type common_reasoning_format = common_reasoning_formatValue<0>|common_reasoning_formatValue<1>|common_reasoning_formatValue<2>|common_reasoning_formatValue<3>;

export interface common_grammar_trigger_typeValue<T extends number> {
  value: T;
}
export type common_grammar_trigger_type = common_grammar_trigger_typeValue<0>|common_grammar_trigger_typeValue<1>|common_grammar_trigger_typeValue<2>|common_grammar_trigger_typeValue<3>;

export interface common_reasoning_budget_stateValue<T extends number> {
  value: T;
}
export type common_reasoning_budget_state = common_reasoning_budget_stateValue<0>|common_reasoning_budget_stateValue<1>|common_reasoning_budget_stateValue<2>|common_reasoning_budget_stateValue<3>|common_reasoning_budget_stateValue<4>;

export interface string_vector extends ClassHandle, Iterable<string> {
  push_back(_0: EmbindString): void;
  resize(_0: number, _1: EmbindString): void;
  size(): number;
  get(_0: number): string | undefined;
  set(_0: number, _1: EmbindString): boolean;
}

export interface llama_tokens extends ClassHandle, Iterable<number> {
  push_back(_0: number): void;
  resize(_0: number, _1: number): void;
  size(): number;
  get(_0: number): number | undefined;
  set(_0: number, _1: number): boolean;
}

export interface llama_token_sequences extends ClassHandle, Iterable<llama_tokens> {
  push_back(_0: llama_tokens): void;
  resize(_0: number, _1: llama_tokens): void;
  size(): number;
  get(_0: number): llama_tokens | undefined;
  set(_0: number, _1: llama_tokens): boolean;
}

export interface size_vector extends ClassHandle, Iterable<bigint> {
  push_back(_0: bigint): void;
  resize(_0: number, _1: bigint): void;
  size(): number;
  get(_0: number): bigint | undefined;
  set(_0: number, _1: bigint): boolean;
}

export interface common_chat_messages extends ClassHandle, Iterable<common_chat_msg> {
  size(): number;
  get(_0: number): common_chat_msg | undefined;
  push_back(_0: common_chat_msg): void;
  resize(_0: number, _1: common_chat_msg): void;
  set(_0: number, _1: common_chat_msg): boolean;
}

export interface common_chat_tools extends ClassHandle, Iterable<common_chat_tool> {
  size(): number;
  get(_0: number): common_chat_tool | undefined;
  push_back(_0: common_chat_tool): void;
  resize(_0: number, _1: common_chat_tool): void;
  set(_0: number, _1: common_chat_tool): boolean;
}

export interface common_chat_tool_calls extends ClassHandle, Iterable<common_chat_tool_call> {
  size(): number;
  get(_0: number): common_chat_tool_call | undefined;
  push_back(_0: common_chat_tool_call): void;
  resize(_0: number, _1: common_chat_tool_call): void;
  set(_0: number, _1: common_chat_tool_call): boolean;
}

export interface common_chat_content_parts extends ClassHandle, Iterable<common_chat_msg_content_part> {
  size(): number;
  get(_0: number): common_chat_msg_content_part | undefined;
  push_back(_0: common_chat_msg_content_part): void;
  resize(_0: number, _1: common_chat_msg_content_part): void;
  set(_0: number, _1: common_chat_msg_content_part): boolean;
}

export interface common_chat_message_diffs extends ClassHandle, Iterable<common_chat_msg_diff> {
  size(): number;
  get(_0: number): common_chat_msg_diff | undefined;
  push_back(_0: common_chat_msg_diff): void;
  resize(_0: number, _1: common_chat_msg_diff): void;
  set(_0: number, _1: common_chat_msg_diff): boolean;
}

export interface common_grammar_triggers extends ClassHandle, Iterable<common_grammar_trigger> {
  size(): number;
  get(_0: number): common_grammar_trigger | undefined;
  push_back(_0: common_grammar_trigger): void;
  resize(_0: number, _1: common_grammar_trigger): void;
  set(_0: number, _1: common_grammar_trigger): boolean;
}

export interface common_chat_delimiter_vector extends ClassHandle, Iterable<common_chat_msg_delimiter> {
  size(): number;
  get(_0: number): common_chat_msg_delimiter | undefined;
  push_back(_0: common_chat_msg_delimiter): void;
  resize(_0: number, _1: common_chat_msg_delimiter): void;
  set(_0: number, _1: common_chat_msg_delimiter): boolean;
}

export interface common_chat_span_vector extends ClassHandle, Iterable<common_chat_msg_span> {
  size(): number;
  get(_0: number): common_chat_msg_span | undefined;
  push_back(_0: common_chat_msg_span): void;
  resize(_0: number, _1: common_chat_msg_span): void;
  set(_0: number, _1: common_chat_msg_span): boolean;
}

export interface string_map extends ClassHandle {
  size(): number;
  get(_0: EmbindString): string | undefined;
  set(_0: EmbindString, _1: EmbindString): void;
  keys(): string_vector;
}

export interface bool_map extends ClassHandle {
  size(): number;
  get(_0: EmbindString): boolean | undefined;
  set(_0: EmbindString, _1: boolean): void;
  keys(): string_vector;
}

export interface size_map extends ClassHandle {
  size(): number;
  get(_0: bigint): bigint | undefined;
  set(_0: bigint, _1: bigint): void;
  keys(): size_vector;
}

export interface common_json extends ClassHandle {
  dump(_0: number): string;
  is_discarded(): boolean;
  is_null(): boolean;
  is_object(): boolean;
  is_array(): boolean;
  size(): bigint;
}

export interface system_clock_duration extends ClassHandle {
  count(): bigint;
}

export interface system_clock_time_point extends ClassHandle {
  time_since_epoch(): system_clock_duration;
}

export interface common_chat_tool_call extends ClassHandle {
  get name(): string;
  set name(value: EmbindString);
  get arguments(): string;
  set arguments(value: EmbindString);
  get id(): string;
  set id(value: EmbindString);
}

export interface common_chat_msg_content_part extends ClassHandle {
  get type(): string;
  set type(value: EmbindString);
  get text(): string;
  set text(value: EmbindString);
}

export interface common_chat_msg extends ClassHandle {
  get role(): string;
  set role(value: EmbindString);
  get content(): string;
  set content(value: EmbindString);
  content_parts: common_chat_content_parts;
  tool_calls: common_chat_tool_calls;
  get reasoning_content(): string;
  set reasoning_content(value: EmbindString);
  get tool_name(): string;
  set tool_name(value: EmbindString);
  get tool_call_id(): string;
  set tool_call_id(value: EmbindString);
  to_json_oaicompat(_0: boolean): common_json;
  render_content(_0: EmbindString): string;
  empty(): boolean;
  contains_media(): boolean;
}

export interface common_chat_tool extends ClassHandle {
  get name(): string;
  set name(value: EmbindString);
  get description(): string;
  set description(value: EmbindString);
  get parameters(): string;
  set parameters(value: EmbindString);
}

export interface common_chat_msg_diff extends ClassHandle {
  get reasoning_content_delta(): string;
  set reasoning_content_delta(value: EmbindString);
  get content_delta(): string;
  set content_delta(value: EmbindString);
  tool_call_index: bigint;
  tool_call_delta: common_chat_tool_call;
}

export interface common_grammar_trigger extends ClassHandle {
  type: common_grammar_trigger_type;
  get value(): string;
  set value(value: EmbindString);
  token: number;
}

export interface common_chat_msg_span extends ClassHandle {
  role: common_chat_role;
  pos: bigint;
  len: bigint;
  valid(): boolean;
}

export interface common_chat_msg_spans extends ClassHandle {
  spans: common_chat_span_vector;
  add(_0: common_chat_role, _1: bigint, _2: bigint): void;
  is_user_start(_0: number): boolean;
  last_user_message_pos(): number;
}

export interface common_chat_msg_delimiter extends ClassHandle {
  role: common_chat_role;
  get delimiter(): string;
  set delimiter(value: EmbindString);
  tokens: llama_tokens;
}

export interface common_chat_msg_delimiters extends ClassHandle {
  delimiters: common_chat_delimiter_vector;
  add(_0: common_chat_role, _1: EmbindString): void;
  tokenize(_0: bigint): void;
  split(_0: llama_tokens, _1: size_map): common_chat_msg_spans;
  to_json(): common_json;
}

export interface common_chat_templates_inputs extends ClassHandle {
  messages: common_chat_messages;
  get grammar(): string;
  set grammar(value: EmbindString);
  get json_schema(): string;
  set json_schema(value: EmbindString);
  add_generation_prompt: boolean;
  continue_final_message: common_chat_continuation;
  use_jinja: boolean;
  tools: common_chat_tools;
  tool_choice: common_chat_tool_choice;
  parallel_tool_calls: boolean;
  reasoning_format: common_reasoning_format;
  enable_thinking: boolean;
  now: system_clock_time_point;
  chat_template_kwargs: string_map;
  add_bos: boolean;
  add_eos: boolean;
  force_pure_content: boolean;
}

export interface common_chat_params extends ClassHandle {
  format: common_chat_format;
  get prompt(): string;
  set prompt(value: EmbindString);
  get grammar(): string;
  set grammar(value: EmbindString);
  grammar_lazy: boolean;
  get generation_prompt(): string;
  set generation_prompt(value: EmbindString);
  supports_thinking: boolean;
  get thinking_start_tag(): string;
  set thinking_start_tag(value: EmbindString);
  thinking_end_tags: string_vector;
  grammar_triggers: common_grammar_triggers;
  preserved_tokens: string_vector;
  additional_stops: string_vector;
  get parser(): string;
  set parser(value: EmbindString);
  message_delimiters: common_chat_msg_delimiters;
}

export interface common_chat_parser_params extends ClassHandle {
  format: common_chat_format;
  reasoning_format: common_reasoning_format;
  reasoning_in_content: boolean;
  get generation_prompt(): string;
  set generation_prompt(value: EmbindString);
  parse_tool_calls: boolean;
  is_continuation: boolean;
  echo: boolean;
  debug: boolean;
  parser: common_peg_arena;
}

export interface common_chat_prompt_preset extends ClassHandle {
  get system(): string;
  set system(value: EmbindString);
  get user(): string;
  set user(value: EmbindString);
}

export interface common_peg_arena extends ClassHandle {
  load(_0: EmbindString): void;
  save(): string;
  empty(): boolean;
  size(): bigint;
  to_json(): common_json;
}

export interface common_chat_templates extends ClassHandle {
  apply(_0: common_chat_templates_inputs): common_chat_params;
  source(_0: EmbindString): string;
  was_explicit(): boolean;
  get_caps(): bool_map;
  support_enable_thinking(): boolean;
  format_single(_0: common_chat_messages, _1: common_chat_msg, _2: boolean, _3: boolean): string;
  format_example(_0: boolean, _1: string_map): string;
  get_asr_prompt(): common_chat_prompt_preset;
}

interface EmbindModule {
  common_chat_role: {COMMON_CHAT_ROLE_UNKNOWN: common_chat_roleValue<0>, COMMON_CHAT_ROLE_SYSTEM: common_chat_roleValue<1>, COMMON_CHAT_ROLE_ASSISTANT: common_chat_roleValue<2>, COMMON_CHAT_ROLE_USER: common_chat_roleValue<3>, COMMON_CHAT_ROLE_TOOL: common_chat_roleValue<4>};
  common_chat_tool_choice: {COMMON_CHAT_TOOL_CHOICE_AUTO: common_chat_tool_choiceValue<0>, COMMON_CHAT_TOOL_CHOICE_REQUIRED: common_chat_tool_choiceValue<1>, COMMON_CHAT_TOOL_CHOICE_NONE: common_chat_tool_choiceValue<2>};
  common_chat_format: {COMMON_CHAT_FORMAT_CONTENT_ONLY: common_chat_formatValue<0>, COMMON_CHAT_FORMAT_PEG_SIMPLE: common_chat_formatValue<1>, COMMON_CHAT_FORMAT_PEG_NATIVE: common_chat_formatValue<2>, COMMON_CHAT_FORMAT_PEG_GEMMA4: common_chat_formatValue<3>, COMMON_CHAT_FORMAT_PEG_MINIMAX_M3: common_chat_formatValue<4>, COMMON_CHAT_FORMAT_COUNT: common_chat_formatValue<5>};
  common_chat_continuation: {COMMON_CHAT_CONTINUATION_NONE: common_chat_continuationValue<0>, COMMON_CHAT_CONTINUATION_AUTO: common_chat_continuationValue<1>, COMMON_CHAT_CONTINUATION_REASONING: common_chat_continuationValue<2>, COMMON_CHAT_CONTINUATION_CONTENT: common_chat_continuationValue<3>};
  common_reasoning_format: {COMMON_REASONING_FORMAT_NONE: common_reasoning_formatValue<0>, COMMON_REASONING_FORMAT_AUTO: common_reasoning_formatValue<1>, COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY: common_reasoning_formatValue<2>, COMMON_REASONING_FORMAT_DEEPSEEK: common_reasoning_formatValue<3>};
  common_grammar_trigger_type: {COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: common_grammar_trigger_typeValue<0>, COMMON_GRAMMAR_TRIGGER_TYPE_WORD: common_grammar_trigger_typeValue<1>, COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: common_grammar_trigger_typeValue<2>, COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: common_grammar_trigger_typeValue<3>};
  common_reasoning_budget_state: {REASONING_BUDGET_IDLE: common_reasoning_budget_stateValue<0>, REASONING_BUDGET_COUNTING: common_reasoning_budget_stateValue<1>, REASONING_BUDGET_FORCING: common_reasoning_budget_stateValue<2>, REASONING_BUDGET_WAITING_UTF8: common_reasoning_budget_stateValue<3>, REASONING_BUDGET_DONE: common_reasoning_budget_stateValue<4>};
  string_vector: {
    new(): string_vector;
  };
  llama_tokens: {
    new(): llama_tokens;
  };
  llama_token_sequences: {
    new(): llama_token_sequences;
  };
  size_vector: {
    new(): size_vector;
  };
  common_chat_messages: {
    new(): common_chat_messages;
  };
  common_chat_tools: {
    new(): common_chat_tools;
  };
  common_chat_tool_calls: {
    new(): common_chat_tool_calls;
  };
  common_chat_content_parts: {
    new(): common_chat_content_parts;
  };
  common_chat_message_diffs: {
    new(): common_chat_message_diffs;
  };
  common_grammar_triggers: {
    new(): common_grammar_triggers;
  };
  common_chat_delimiter_vector: {
    new(): common_chat_delimiter_vector;
  };
  common_chat_span_vector: {
    new(): common_chat_span_vector;
  };
  string_map: {
    new(): string_map;
  };
  bool_map: {
    new(): bool_map;
  };
  size_map: {
    new(): size_map;
  };
  common_json: {
    new(): common_json;
    parse(_0: EmbindString): common_json;
    parse_no_throw(_0: EmbindString): common_json;
    array(): common_json;
    object(): common_json;
  };
  system_clock_duration: {
    new(): system_clock_duration;
    new(_0: bigint): system_clock_duration;
  };
  system_clock_time_point: {
    new(): system_clock_time_point;
    new(_0: system_clock_duration): system_clock_time_point;
  };
  system_clock_period_num(): bigint;
  system_clock_period_den(): bigint;
  system_clock_now(): system_clock_time_point;
  common_chat_tool_call: {
    new(): common_chat_tool_call;
  };
  common_chat_msg_content_part: {
    new(): common_chat_msg_content_part;
  };
  common_chat_msg: {
    new(): common_chat_msg;
  };
  common_chat_tool: {
    new(): common_chat_tool;
  };
  common_chat_msg_diff: {
    new(): common_chat_msg_diff;
    compute_diffs(_0: common_chat_msg, _1: common_chat_msg): common_chat_message_diffs;
  };
  common_chat_no_tool_call_index(): bigint;
  common_grammar_trigger: {
    new(): common_grammar_trigger;
  };
  common_chat_msg_span: {
    new(): common_chat_msg_span;
  };
  common_chat_msg_spans: {
    new(): common_chat_msg_spans;
  };
  common_chat_msg_delimiter: {
    new(): common_chat_msg_delimiter;
  };
  common_chat_msg_delimiters: {
    new(): common_chat_msg_delimiters;
  };
  common_chat_templates_inputs: {
    new(): common_chat_templates_inputs;
  };
  common_chat_params: {
    new(): common_chat_params;
  };
  common_chat_parser_params: {
    new(): common_chat_parser_params;
    new(_0: common_chat_params): common_chat_parser_params;
  };
  common_chat_prompt_preset: {
    new(): common_chat_prompt_preset;
  };
  common_peg_arena: {
    new(): common_peg_arena;
    from_json(_0: common_json): common_peg_arena;
  };
  common_chat_templates: {
    new(_0: bigint, _1: EmbindString, _2: EmbindString, _3: EmbindString): common_chat_templates;
  };
  common_chat_verify_template(_0: EmbindString, _1: boolean): boolean;
  common_chat_parse(_0: EmbindString, _1: boolean, _2: common_chat_parser_params): common_chat_msg;
  common_chat_peg_parse(_0: common_peg_arena, _1: EmbindString, _2: boolean, _3: common_chat_parser_params): common_chat_msg;
  common_chat_msgs_parse_oaicompat(_0: common_json): common_chat_messages;
  common_chat_msgs_to_json_oaicompat(_0: common_chat_messages, _1: boolean): common_json;
  common_chat_tools_parse_oaicompat(_0: common_json): common_chat_tools;
  common_chat_tools_to_json_oaicompat(_0: common_chat_tools): common_json;
  common_chat_continuation_parse(_0: common_json): common_chat_continuation;
  common_chat_tool_choice_parse_oaicompat(_0: EmbindString): common_chat_tool_choice;
  common_chat_msg_delimiters_parse(_0: common_json): common_chat_msg_delimiters;
  common_chat_role_from_string(_0: EmbindString): common_chat_role;
  common_chat_role_to_string(_0: common_chat_role): string;
  common_chat_format_name(_0: common_chat_format): string;
  common_reasoning_format_name(_0: common_reasoning_format): string;
  common_reasoning_format_from_name(_0: EmbindString): common_reasoning_format;
  json_schema_to_grammar(_0: common_json, _1: boolean): string;
  common_reasoning_budget_init(_0: bigint, _1: llama_token_sequences, _2: llama_token_sequences, _3: llama_tokens, _4: number, _5: common_reasoning_budget_state): bigint;
  common_reasoning_budget_get_state(_0: bigint): common_reasoning_budget_state;
  common_reasoning_budget_get_end_match_copy(_0: bigint): llama_tokens;
  common_reasoning_budget_force(_0: bigint): boolean;
}

export type MainModule = WasmModule & typeof RuntimeExports & EmbindModule;
export default function MainModuleFactory (options?: unknown): Promise<MainModule>;

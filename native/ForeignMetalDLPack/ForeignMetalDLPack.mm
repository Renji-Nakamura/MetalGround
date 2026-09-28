
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Minimal legacy DLPack ABI sufficient for a float32 Metal tensor.
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLMetal = 8
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

typedef struct {
    PyObject_HEAD
    id<MTLBuffer> buffer;
    int32_t ndim;
    int64_t* shape;
    int64_t* strides;
    int64_t size;
} ForeignMetalTensor;

static void dlpack_managed_deleter(DLManagedTensor* self) {
    if (!self) return;
    id<MTLBuffer> buffer = (id<MTLBuffer>)self->manager_ctx;
    if (buffer) {
        [buffer release];
    }
    free(self->dl_tensor.shape);
    free(self->dl_tensor.strides);
    free(self);
}

static void dlpack_capsule_destructor(PyObject* capsule) {
    // DLPack consumers rename a consumed capsule to "used_dltensor".
    // Only free an unconsumed "dltensor".
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor* managed =
            (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
        if (managed && managed->deleter) {
            managed->deleter(managed);
        }
    }
}

static void ForeignMetalTensor_dealloc(ForeignMetalTensor* self) {
    if (self->buffer) {
        [self->buffer release];
        self->buffer = nil;
    }
    free(self->shape);
    free(self->strides);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* ForeignMetalTensor_dlpack_device(
    ForeignMetalTensor* self, PyObject* Py_UNUSED(ignored)) {
    return Py_BuildValue("(ii)", (int)kDLMetal, 0);
}

static PyObject* ForeignMetalTensor_dlpack(
    ForeignMetalTensor* self, PyObject* args, PyObject* kwargs) {
    // Accept and intentionally ignore any modern __dlpack__ negotiation
    // keywords. The legacy dltensor capsule is sufficient for this probe.
    (void)args;
    (void)kwargs;

    DLManagedTensor* managed =
        (DLManagedTensor*)calloc(1, sizeof(DLManagedTensor));
    if (!managed) return PyErr_NoMemory();

    managed->dl_tensor.shape =
        (int64_t*)malloc(sizeof(int64_t) * self->ndim);
    managed->dl_tensor.strides =
        (int64_t*)malloc(sizeof(int64_t) * self->ndim);
    if (!managed->dl_tensor.shape || !managed->dl_tensor.strides) {
        free(managed->dl_tensor.shape);
        free(managed->dl_tensor.strides);
        free(managed);
        return PyErr_NoMemory();
    }

    memcpy(
        managed->dl_tensor.shape,
        self->shape,
        sizeof(int64_t) * self->ndim
    );
    memcpy(
        managed->dl_tensor.strides,
        self->strides,
        sizeof(int64_t) * self->ndim
    );

    [self->buffer retain];
    managed->manager_ctx = (void*)self->buffer;
    managed->deleter = dlpack_managed_deleter;

    managed->dl_tensor.data = (void*)self->buffer;  // MTLBuffer*, not contents()
    managed->dl_tensor.device = {kDLMetal, 0};
    managed->dl_tensor.ndim = self->ndim;
    managed->dl_tensor.dtype = {
        (uint8_t)kDLFloat, 32, 1
    };
    managed->dl_tensor.byte_offset = 0;

    PyObject* capsule = PyCapsule_New(
        managed, "dltensor", dlpack_capsule_destructor
    );
    if (!capsule) {
        managed->deleter(managed);
        return nullptr;
    }
    return capsule;
}

static PyObject* ForeignMetalTensor_read(
    ForeignMetalTensor* self, PyObject* args) {
    long long index = 0;
    if (!PyArg_ParseTuple(args, "L", &index)) return nullptr;
    if (index < 0 || index >= self->size) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return nullptr;
    }
    float* p = (float*)[self->buffer contents];
    return PyFloat_FromDouble((double)p[index]);
}

static PyObject* ForeignMetalTensor_write(
    ForeignMetalTensor* self, PyObject* args) {
    long long index = 0;
    double value = 0.0;
    if (!PyArg_ParseTuple(args, "Ld", &index, &value)) return nullptr;
    if (index < 0 || index >= self->size) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return nullptr;
    }
    float* p = (float*)[self->buffer contents];
    p[index] = (float)value;
    Py_RETURN_NONE;
}

static PyObject* ForeignMetalTensor_storage_mode(
    ForeignMetalTensor* self, PyObject* Py_UNUSED(ignored)) {
    return PyLong_FromLong((long)[self->buffer storageMode]);
}

static PyObject* ForeignMetalTensor_length_bytes(
    ForeignMetalTensor* self, PyObject* Py_UNUSED(ignored)) {
    return PyLong_FromUnsignedLongLong(
        (unsigned long long)[self->buffer length]
    );
}

static PyObject* ForeignMetalTensor_shape(
    ForeignMetalTensor* self, PyObject* Py_UNUSED(ignored)) {
    PyObject* tup = PyTuple_New(self->ndim);
    if (!tup) return nullptr;
    for (int i = 0; i < self->ndim; ++i) {
        PyObject* v = PyLong_FromLongLong(self->shape[i]);
        if (!v) {
            Py_DECREF(tup);
            return nullptr;
        }
        PyTuple_SET_ITEM(tup, i, v);
    }
    return tup;
}

static PyMethodDef ForeignMetalTensor_methods[] = {
    {
        "__dlpack_device__",
        (PyCFunction)ForeignMetalTensor_dlpack_device,
        METH_NOARGS,
        nullptr
    },
    {
        "__dlpack__",
        (PyCFunction)(void(*)(void))ForeignMetalTensor_dlpack,
        METH_VARARGS | METH_KEYWORDS,
        nullptr
    },
    {"read", (PyCFunction)ForeignMetalTensor_read, METH_VARARGS, nullptr},
    {"write", (PyCFunction)ForeignMetalTensor_write, METH_VARARGS, nullptr},
    {
        "storage_mode",
        (PyCFunction)ForeignMetalTensor_storage_mode,
        METH_NOARGS,
        nullptr
    },
    {
        "length_bytes",
        (PyCFunction)ForeignMetalTensor_length_bytes,
        METH_NOARGS,
        nullptr
    },
    {"shape", (PyCFunction)ForeignMetalTensor_shape, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

static PyType_Slot ForeignMetalTensor_slots[] = {
    {Py_tp_dealloc, (void*)ForeignMetalTensor_dealloc},
    {Py_tp_methods, (void*)ForeignMetalTensor_methods},
    {0, nullptr}
};

static PyType_Spec ForeignMetalTensor_spec = {
    "metal_dlpack_native.ForeignMetalTensor",
    sizeof(ForeignMetalTensor),
    0,
    Py_TPFLAGS_DEFAULT,
    ForeignMetalTensor_slots
};

static PyObject* ForeignMetalTensorType = nullptr;

static PyObject* make_zeros(PyObject* module, PyObject* args) {
    PyObject* shape_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) return nullptr;

    PyObject* seq = PySequence_Fast(shape_obj, "shape must be a sequence");
    if (!seq) return nullptr;

    Py_ssize_t ndim_ss = PySequence_Fast_GET_SIZE(seq);
    if (ndim_ss <= 0 || ndim_ss > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "shape rank must be 1..8");
        return nullptr;
    }

    int32_t ndim = (int32_t)ndim_ss;
    int64_t* shape = (int64_t*)malloc(sizeof(int64_t) * ndim);
    int64_t* strides = (int64_t*)malloc(sizeof(int64_t) * ndim);
    if (!shape || !strides) {
        free(shape);
        free(strides);
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }

    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        long long dim = PyLong_AsLongLong(item);
        if (PyErr_Occurred() || dim <= 0) {
            free(shape);
            free(strides);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "all shape dims must be > 0");
            return nullptr;
        }
        if (size > INT64_MAX / dim) {
            free(shape);
            free(strides);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_OverflowError, "tensor too large");
            return nullptr;
        }
        shape[i] = (int64_t)dim;
        size *= (int64_t)dim;
    }
    Py_DECREF(seq);

    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    NSUInteger nbytes = (NSUInteger)(size * (int64_t)sizeof(float));

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        [pool drain];
        free(shape);
        free(strides);
        PyErr_SetString(PyExc_RuntimeError, "No Metal device");
        return nullptr;
    }

    id<MTLBuffer> buffer = [
        device newBufferWithLength:nbytes
        options:MTLResourceStorageModeShared
    ];
    if (!buffer) {
        [pool drain];
        free(shape);
        free(strides);
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate shared MTLBuffer");
        return nullptr;
    }
    memset([buffer contents], 0, nbytes);

    ForeignMetalTensor* obj = (ForeignMetalTensor*)PyObject_CallNoArgs(
        ForeignMetalTensorType
    );
    if (!obj) {
        [buffer release];
        [pool drain];
        free(shape);
        free(strides);
        return nullptr;
    }

    obj->buffer = buffer; // owns +1 from newBuffer...
    obj->ndim = ndim;
    obj->shape = shape;
    obj->strides = strides;
    obj->size = size;

    [pool drain];
    return (PyObject*)obj;
}

static PyMethodDef module_methods[] = {
    {
        "make_zeros",
        (PyCFunction)make_zeros,
        METH_VARARGS,
        "Allocate a shared MTLBuffer-backed float32 tensor."
    },
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "metal_dlpack_native",
    nullptr,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_metal_dlpack_native(void) {
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) return nullptr;

    ForeignMetalTensorType = PyType_FromSpec(&ForeignMetalTensor_spec);
    if (!ForeignMetalTensorType) {
        Py_DECREF(module);
        return nullptr;
    }

    if (PyModule_AddObject(
            module,
            "ForeignMetalTensor",
            ForeignMetalTensorType) < 0) {
        Py_DECREF(ForeignMetalTensorType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}

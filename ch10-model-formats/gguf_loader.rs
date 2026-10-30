use std::collections::HashMap;
use std::io::{Read, Seek};

/// Parse a GGUF file header and extract tensor metadata.
pub struct GgufParser {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

impl GgufParser {
    /// Read the GGUF header from a file.
    ///
    /// The header is 24 bytes (magic + version + tensor_count +
    /// metadata_count). `parse_metadata`/`parse_tensor_headers` operate on
    /// your own `File` handle - seek it to offset 24 before calling them.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;

        // Read magic number (4 bytes)
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;

        if magic != *b"GGUF" {
            return Err("Invalid GGUF magic number".into());
        }

        // Read version (4 bytes, little-endian)
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != 2 && version != 3 {
            return Err(format!("Unsupported GGUF version: {}", version).into());
        }

        // Read tensor count (8 bytes, little-endian)
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes);

        // Read metadata KV pair count (8 bytes, little-endian)
        let mut metadata_count_bytes = [0u8; 8];
        file.read_exact(&mut metadata_count_bytes)?;
        let metadata_count = u64::from_le_bytes(metadata_count_bytes);

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_count,
        })
    }

    /// Parse the metadata KV pairs (returns the string values).
    ///
    /// Non-string values are skipped over; string values (type 8) are
    /// captured. GGUF strings are a u64 length followed by exactly that
    /// many UTF-8 bytes - they are NOT NUL-terminated. (We still trim a
    /// stray trailing NUL defensively, in case a file was written by an
    /// early tool that padded.)
    pub fn parse_metadata(&mut self, file: &mut std::fs::File) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();

        for _ in 0..self.metadata_count {
            // Read key: u64 length + that many bytes
            let key = read_gguf_string(file)?;

            // Read value type (4-byte little-endian enum, NOT 1 byte)
            let value_type = read_u32(file)?;

            // Read value based on type (0=uint8, 1=int8, 2=uint16,
            // 3=int16, 4=uint32, 5=int32, 6=f32, 7=bool, 8=string,
            // 9=array, 10=uint64, 11=int64, 12=f64)
            if value_type == 8 {
                metadata.insert(key, read_gguf_string(file)?);
            } else {
                skip_gguf_value(file, value_type)?;
            }
        }

        Ok(metadata)
    }

    /// Parse tensor headers and return their metadata.
    pub fn parse_tensor_headers(&mut self, file: &mut std::fs::File) -> Result<Vec<GgufTensorInfo>, Box<dyn std::error::Error>> {
        let mut tensors = Vec::new();

        for _ in 0..self.tensor_count {
            // Read tensor name (u64 length + bytes; at most 64 bytes in v3)
            let name = read_gguf_string(file)?;

            // Read number of dimensions (4 bytes; at most 4 in v3)
            let ndim = read_u32(file)? as usize;

            // Read shape (8 bytes per dimension, little-endian)
            let mut shape = vec![0u64; ndim];
            for s in &mut shape {
                let mut buf = [0u8; 8];
                file.read_exact(&mut buf)?;
                *s = u64::from_le_bytes(buf);
            }

            // Read quantization type (4 bytes: a ggml_type enum)
            let quant_type = read_u32(file)?;

            // Read data offset (8 bytes, little-endian). Note: this is
            // RELATIVE TO THE START OF THE DATA SECTION (after the
            // alignment padding), not to the start of the file.
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let data_offset = u64::from_le_bytes(offset_bytes);

            tensors.push(GgufTensorInfo {
                name,
                shape,
                quant_type,
                data_offset,
            });
        }

        Ok(tensors)
    }
}

/// Metadata for a single GGUF tensor.
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub quant_type: u32,
    pub data_offset: u64,
}

fn read_u32(file: &mut std::fs::File) -> Result<u32, Box<dyn std::error::Error>> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a GGUF string: u64 length + exactly that many UTF-8 bytes
/// (no NUL terminator in the format).
fn read_gguf_string(file: &mut std::fs::File) -> Result<String, Box<dyn std::error::Error>> {
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?.trim_end_matches('\0').to_string())
}

/// Skip a GGUF value (scalars, strings and arrays) without decoding it.
/// Value types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
/// 6=float32, 7=bool, 8=string, 9=array (recursed into below),
/// 10=uint64, 11=int64, 12=float64.
fn skip_gguf_value(file: &mut std::fs::File, value_type: u32) -> Result<(), Box<dyn std::error::Error>> {
    match value_type {
        0 | 1 | 7 => { file.seek(std::io::SeekFrom::Current(1))?; Ok(()) }  // 1 byte
        2 | 3 => { file.seek(std::io::SeekFrom::Current(2))?; Ok(()) }      // 2 bytes
        4..=6 => { file.seek(std::io::SeekFrom::Current(4))?; Ok(()) }      // 4 bytes
        10..=12 => { file.seek(std::io::SeekFrom::Current(8))?; Ok(()) }     // 8 bytes
        8 => { read_gguf_string(file)?; Ok(()) }
        9 => {
            // Array: u32 element type, u64 count, then count elements
            // (which may themselves be strings or arrays)
            let element = read_u32(file)?;
            let mut count_bytes = [0u8; 8];
            file.read_exact(&mut count_bytes)?;
            for _ in 0..u64::from_le_bytes(count_bytes) {
                skip_gguf_value(file, element)?;
            }
            Ok(())
        }
        other => Err(format!("Unsupported GGUF value type: {}", other).into()),
    }
}

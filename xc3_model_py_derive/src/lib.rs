extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;

use syn::{parse_macro_input, Data, DataStruct, DeriveInput, Fields};

#[proc_macro_derive(MapPy, attributes(map))]
pub fn map_py_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // ex: #[map(xc3_model::Material)]
    let map_type: syn::Path = input
        .attrs
        .iter()
        .find(|a| a.path().is_ident("map"))
        .map(|a| a.parse_args().unwrap())
        .expect("Must specify a map type");

    let name = &input.ident;

    // Assume both structs have identical field names.
    // This could be improved via skip and rename attributes in the future.
    let map_fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => {
            let named_fields: Vec<_> = fields.named.iter().map(|field| &field.ident).collect();
            quote! {
                #(
                    #named_fields: self.#named_fields.map_py(py)?
                ),*
            }
        }
        _ => panic!("Unsupported type"),
    };

    quote! {
        // Map from the implementing type to the map type.
        impl crate::MapPy<#map_type> for #name {
            fn map_py(self, py: pyo3::Python) -> pyo3::prelude::PyResult<#map_type> {
                Ok(
                    #map_type {
                        #map_fields
                    }
                )
            }
        }

        // Map from the map type to the implementing type.
        impl crate::MapPy<#name> for #map_type {
            fn map_py(self, py: pyo3::Python) -> pyo3::prelude::PyResult<#name> {
                Ok(
                    #name {
                        #map_fields
                    }
                )
            }
        }

        // Map to and from Py<T>
        impl crate::MapPy<Py<#name>> for #map_type {
            fn map_py(self, py: Python) -> PyResult<Py<#name>> {
                let value: #name = self.map_py(py)?;
                Py::new(py, value)
            }
        }
    }
    .into()
}

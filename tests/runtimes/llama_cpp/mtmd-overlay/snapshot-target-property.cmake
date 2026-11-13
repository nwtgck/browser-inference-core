# Test-only snapshot of a target property before/after applying the overlays.
# get_target_property() uses <output-variable>-NOTFOUND for an unset property;
# those variable-specific sentinels are not values that can be compared.
function(lcb_snapshot_target_property target property output)
    get_property(property_is_set TARGET "${target}" PROPERTY "${property}" SET)
    if(property_is_set)
        get_property(property_value TARGET "${target}" PROPERTY "${property}")
    else()
        set(property_value "")
    endif()
    # Preserve unset versus explicitly empty, as well as the exact list/order.
    # Never test property_value as a boolean: OFF and *-NOTFOUND can be values.
    file(WRITE "${output}" "${property_is_set}\n${property_value}\n")
endfunction()

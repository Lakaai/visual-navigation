#include <eigen3/Eigen/Core>

constexpr double eccentricity_squared = { 6.69437999014e-3 };
constexpr double semi_major_axis = { 6378137.0 };
// def lla2ecef(lat, lon, alt, latlon_unit='deg', alt_unit='m', model='wgs84'):
//     """
//     Convert Latitude, Longitude, Altitude, to ECEF position
    
//     Parameters
//     ----------
//     lat : {(N,)} array like latitude, unit specified by latlon_unit, default in deg
//     lon : {(N,)} array like longitude, unit specified by latlon_unit, default in deg
//     alt : {(N,)} array like altitude, unit specified by alt_unit, default in m
    
//     Returns
//     -------
//     ecef : {(N,3)} array like ecef position, unit is the same as alt_unit
//     """
//     lat,N1 = _input_check_Nx1(lat)
//     lon,N2 = _input_check_Nx1(lon)
//     alt,N3 = _input_check_Nx1(alt)
    
//     if( (N1!=N2) or (N2!=N3) or (N1!=N3) ):
//         raise ValueError('Inputs are not of the same dimension')
    
//     if(model=='wgs84'):
//         Rew,Rns = earthrad(lat,lat_unit=latlon_unit)
//     else:
//         Rew = wgs84.a 
    
//     if(latlon_unit=='deg'):
//         lat = np.deg2rad(lat)
//         lon = np.deg2rad(lon)
    
//     x = (Rew + alt)*np.cos(lat)*np.cos(lon)
//     y = (Rew + alt)*np.cos(lat)*np.sin(lon)
//     z = ( (1-wgs84._ecc_sqrd)*Rew + alt )*np.sin(lat)
    
//     ecef = np.vstack((x,y,z)).T

//     if(N1==1):
//         ecef = ecef.reshape(3)

//     return ecef

/// @brief 
/// @param lat degrees
/// @param lon degrees
/// @param alt 
/// @return 
static Eigen::Vector3d lla2ecef(double lat, double lon, double alt)
{
    lat = lat * M_PI / 180.0;
    lon = lon * M_PI / 180.0;

    auto N = [](double lat) {
        return semi_major_axis /
               sqrt(1.0 - eccentricity_squared * sin(lat) * sin(lat));
    };

    double N_phi = N(lat);

    double x = (N_phi + alt) * cos(lat) * cos(lon);
    double y = (N_phi + alt) * cos(lat) * sin(lon);
    double z = (N_phi * (1.0 - eccentricity_squared) + alt) * sin(lat);

    return Eigen::Vector3d(x, y, z);
}


/// @brief 
/// @param ecef
/// @param lat_ref 
/// @param lon_ref
/// @param alt_ref 
/// @return
/// 
static Eigen::Vector3d ecef2ned(Eigen::Vector3d ecef, double lat_ref, double lon_ref, double alt_ref)
{
    lat_ref = lat_ref * M_PI / 180.0;
    lon_ref = lon_ref * M_PI / 180.0;
    Eigen::Matrix3d C;  

    C(0,0) = -sin(lat_ref)*cos(lon_ref);
    C(0,1) = -sin(lat_ref)*sin(lon_ref);
    C(0,2) =  cos(lat_ref);

    C(1,0) = -sin(lon_ref);
    C(1,1) =  cos(lon_ref);
    C(1,2) =  0;

    C(2,0) = -cos(lat_ref)*cos(lon_ref);
    C(2,1) = -cos(lat_ref)*sin(lon_ref);
    C(2,2) = -sin(lat_ref);

    Eigen::Vector3d ned = C * ecef;
    
    return ned;
}

static Eigen::Vector3d lla2ned(
    double lat, double lon, double alt,
    double lat_ref, double lon_ref, double alt_ref)
{
    Eigen::Vector3d ecef     = lla2ecef(lat, lon, alt);
    Eigen::Vector3d ecef_ref = lla2ecef(lat_ref, lon_ref, alt_ref);

    Eigen::Vector3d delta = ecef - ecef_ref;

    lat_ref *= M_PI / 180.0;
    lon_ref *= M_PI / 180.0;

    Eigen::Matrix3d C;

    C(0,0) = -sin(lat_ref)*cos(lon_ref);
    C(0,1) = -sin(lat_ref)*sin(lon_ref);
    C(0,2) =  cos(lat_ref);

    C(1,0) = -sin(lon_ref);
    C(1,1) =  cos(lon_ref);
    C(1,2) =  0;

    C(2,0) = -cos(lat_ref)*cos(lon_ref);
    C(2,1) = -cos(lat_ref)*sin(lon_ref);
    C(2,2) = -sin(lat_ref);

    return C * delta;
}

// def earthrad(lat, lat_unit='deg', model='wgs84'):
//     """
//     Calculate radius of curvature in the prime vertical (East-West) and 
//     meridian (North-South) at a given latitude.

//     Parameters
//     ----------
//     lat : {(N,)} array like latitude, unit specified by lat_unit, default in deg
    
//     Returns
//     -------
//     R_N : {(N,)} array like, radius of curvature in the prime vertical (East-West)
//     R_M : {(N,)} array like, radius of curvature in the meridian (North-South)

//     """
//     if(lat_unit=='deg'):
//         lat = np.deg2rad(lat)
//     elif(lat_unit=='rad'):
//         pass
//     else:
//         raise ValueError('Input unit unknown')

//     if(model=='wgs84'):
//         R_N = wgs84.a/(1-wgs84._ecc_sqrd*np.sin(lat)**2)**0.5
//         R_M = wgs84.a*(1-wgs84._ecc_sqrd)/(1-wgs84._ecc_sqrd*np.sin(lat)**2)**1.5
//     else:
//         raise ValueError('Model unknown')
    
//     return R_N, R_M


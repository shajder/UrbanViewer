#ifndef SHADERS_H
#define SHADERS_H 1

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

const char * terrain_filter_vert =
R"(
out vec4 vcolor;                                                        
out vec4 ecp;                                                           
out vec3 normal;                                                        
out vec2 tcw;                                                           
                                                                            
uniform vec3 lightPos;                                                      
uniform vec4 lightAmbient;                                                  
uniform vec4 lightDiffuse;                                                  
uniform mat4 osg_ViewMatrixInverse;                                         
                                                                            
vec4 vert_proc(in vec4 vert, in vec3 ecnor)                                 
{                                                                           
  vec4 ambiCol = vec4(0.0);                                                 
  vec4 diffCol = vec4(0.0);                                                 
  float nDotVP    = max(0.0, dot(ecnor, lightPos));                         
                                                                            
  ambiCol  = lightAmbient;                                                  
  diffCol  = lightDiffuse * nDotVP;                                         
  vcolor = (ambiCol + diffCol);                                             
                                                                            
  normal = vec3(0.0, 0.0, 1.0);                                                       
  ecp = gl_ModelViewMatrix * vert;                                          
  tcw = vert.xy * 0.001;                                                    
  return vert;                                                              
})";

////////////////////////////////////////////////////////////////////////////////

const char * tiled_terrain_color_filter_PSSM_frag =
R"(
#ifdef _ANDROID_
precision mediump float;                                                                 
uniform highp vec3 lightPos;                                                             
uniform highp mat4 osg_ViewMatrix;                                                       
in highp vec4 ecp;                                                                  
in highp vec2 tcw;                                                                  
#else
uniform vec3 lightPos;                                                                   
uniform mat4 osg_ViewMatrix;                                                             
in vec4 ecp;                                                                        
in vec2 tcw;                                                                        
#endif
                                                                                         
in vec3 normal;                                                                     
uniform sampler2D sampler1;                                                              
uniform sampler2D sampler2;                                                              
uniform sampler2D sampler3;                                                              
                                                                                         
const float fresnel_approx_pow_factor = 2.0;
const float dyna_range = 0.8f;
const vec3 water_bright = vec3(0.5, 1.6, 2.15);
const vec3 water_dark = vec3(0.03, 0.06, 0.135);
const float exposure = 0.4;
                                                                                         
in vec4 vcolor;                                                                     
                                                                                         
vec4 ColorFilter( in vec4 color )                                                        
{                                                                                        
	vec4 c0 = texture2D(sampler1, tcw*30.0) * vcolor;                                       
	vec4 c1 = texture2D(sampler2, tcw*20.0) * vcolor;                                       
	vec4 c2 = vec4(1.0);                                                                    
                                                                                         
  vec3 NH = texture2D(sampler3, tcw*0.5).xyz * vec3(2.0) - vec3(1.0);                    
  vec3 N = normal;                                                                       
  vec3 T = normalize(cross(vec3(0, 1, 0), N));                                           
                                                                                         
  mat3 vmo = mat3(osg_ViewMatrix[0].xyz, osg_ViewMatrix[1].xyz, osg_ViewMatrix[2].xyz);  
                                                                                         
  N = normalize(vmo * N);                                                                
  T = normalize(vmo * T);                                                                
  vec3 B = normalize(cross(N, T));                                                       
                                                                                         
  mat3 tbn = mat3(T, B, N);                                                              
  vec3 ecNorm = N;                                                                       
                                                                                         
  vec3 ecPosNorm = normalize(ecp.xyz);                                                   
  float NdotE = dot(N, -ecPosNorm);                                                      
                                                                                         
  N = tbn * NH;                                                                          
  N = mix( N, ecNorm, pow(abs(1.0-NdotE), 3.0));                                         
  vec3 L = lightPos;                                                                     
  vec3 specular = vec3(0.0);                                                             
  float nDotVP = max(0.0, dot(N, L));                                                    
                                                                                         
  if (nDotVP > 0.0)                                                                      
  {                                                                                      
    vec3 E = -ecPosNorm;                                                                 
    vec3 R = normalize(reflect(-L, N));                                                  
                                                                                         
    float specPow = 72.0;                                                                
    float specScl = 0.4;                                                                 
    specular = vec3(pow(max(dot(R, E), 0.0), specPow) * specScl);                        
                                                                                         
    specular.rgb *= max(0.0, 1.0-pow(abs(1.0-NdotE), 8.0));                              
  }                                                                                      
  NdotE = dot(N, -ecPosNorm);  

  float fresnel = clamp(pow( 1.0 + NdotE, -fresnel_approx_pow_factor ) * dyna_range, 0.0, 1.0);
  vec3 bright = fresnel * water_bright;
  vec3 water = (1.0 - fresnel) * water_dark * water_bright * nDotVP;
  vec4 water_color = vec4(bright + water + specular, 1.0);

  c2 = water_color;                                                                           
                                                                                         
  const float eps = 0.05;                                                                
  float f0 = 0.0;                                                                        
  float f1 = 0.0;                                                                        
  float f2 = 0.0;                                                                        
  if ( (color.b - 0.0) < eps ) f0 = 1.0;                                                 
  else if ( (color.b - 0.5) < eps ) f1 = 1.0;                                            
  else if ( (color.b - 1.0) < eps ) f2 = 1.0;                                            
	 vec4 cult = c0*vec4(f0)+c1*vec4(f1)+c2*vec4(f2);                                       
                                                                                         
  float road_fade_fac = (sign(0.01 - color.r) - 1.0) * (-0.5);                           
  float road_fade_val = 0.2+0.8*(1.0 - pow(abs(1.0-color.r),4.0));                       
	 cult.rgb *= mix(vec3(1.0), vec3(road_fade_val), road_fade_fac);                        
	 vec4 ocol = cult;                                                                      
  return ocol;                                                                           
})";

////////////////////////////////////////////////////////////////////////////////

const char * terrain_color_filter_PSSM_frag =
R"(
uniform sampler2D sampler1;                   
uniform sampler2D sampler3;                   
uniform vec2 viewport;                        
uniform vec4 lightAmbient;                    
                                              
varying float fade;                           
varying vec4 vcolor;                          
varying vec4 ecp;                             
                                              
vec4 ColorFilter( in vec4 color )             
{                                             
	   vec4 ocol = color * vcolor;               
    return ocol;                              
})";

////////////////////////////////////////////////////////////////////////////////

const char * terrain_color_filter_vert =
R"(
varying vec4 vcolor;                                             
varying vec4 ecp;                                                
                                                                 
uniform vec3 lightPos;                                           
uniform vec4 lightAmbient;                                       
uniform vec4 lightDiffuse;                                       
                                                                 
vec4 vert_proc(in vec4 vert, in vec3 ecnor)                      
{                                                                
  vec4 ambiCol = vec4(0.0);                                      
  vec4 diffCol = vec4(0.0);                                      
  float nDotVP    = max(0.0, dot(ecnor, lightPos));              
                                                                 
  ambiCol  = lightAmbient;                                       
  diffCol  = lightDiffuse * nDotVP;                              
  vcolor = (ambiCol + diffCol);                                  
                                                                 
  ecp = gl_ModelViewMatrix * vert;                               
  return vert;                                                   
})";


////////////////////////////////////////////////////////////////////////////////

const char * layered_shadow_cast_geom =
R"(#version 430 compatibility                                 
layout (triangles) in;                                     
layout(triangle_strip, max_vertices = 24) out;             
in vec4 ecPos[];                                           
                                                           
vec4 ppPos[3];                                             
                                                           
void main()                                                
{                                                          
  for (int j = 0; j < NUM_LAYERS; j++)                     
  {                                                        
    bool inRange=false;                                    
    for (int i = 0; i < 3; i++)                            
    {                                                      
      ppPos[i] = projMat[j] * ecPos[i];                    
      vec4 app = abs(ppPos[i]);                            
      if(app.x<1.0&&app.y<1.0&&app.z<1.0)                  
        inRange=true;                                      
    }                                                      
    if ( inRange==true)                                    
    {                                                      
      for (int i = 0; i < 3; i++)                          
      {                                                    
        gl_Layer = j;                                      
        gl_PrimitiveID = j;                                
        gl_Position = ppPos[i];                            
        EmitVertex();                                      
      }                                                    
      EndPrimitive();                                      
    }                                                      
  }                                                        
})";

////////////////////////////////////////////////////////////////////////////////

const char * screen_vert =
R"(#version 420 compatibility
out vec2 tc;                                                       
void main( )                                                           
{                                                                      
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;     
    tc = gl_MultiTexCoord0.xy;                                              
})";

////////////////////////////////////////////////////////////////////////////////

const char * screen_frag =
R"(#version 420 compatibility                                       
#ifdef _ANDROID_
precision mediump float;                                         
#endif                                                           
uniform sampler2D sampler0;                                      
uniform sampler2D sampler1;                                      
                                                                 
layout(location = 0, index = 0) out vec4 out_Color;              
layout(location = 1, index = 0) out float out_Depth;             
                                                                 
in vec2 tc;                                                 
                                                                 
void main (void)                                                 
{                                                                
  vec4 color0 = texture2D(sampler0, tc);                         
  out_Color = color0;                                            
  out_Depth = texture2D(sampler1, tc).x;                         
})";

////////////////////////////////////////////////////////////////////////////////

const char * screen_linearize_frag =
R"(#version 430 compatibility                                                   
#ifdef _ANDROID_
precision mediump float;                                                     
#endif                                                                                       
uniform vec4 clipInfo;                                                       
uniform sampler2D sampler0;                                                  
layout(location = 0, index = 0) out float out_Color;                         
                                                                             
float reconstructCSZ(float d, vec4 clipInfo) 
{                               
    return ((clipInfo[0]*clipInfo[2]) / (clipInfo[1] * d + clipInfo[2]));    
}                                                                            
                                                                             
void main() 
{                                                                
  float depth = texelFetch(sampler0, ivec2(gl_FragCoord.xy), 0).x;           
  out_Color = reconstructCSZ(depth, clipInfo);                               
})";


////////////////////////////////////////////////////////////////////////////////

const char *screen_HBAO_frag =
R"(#version 430 compatibility
#ifdef _ANDROID_
precision mediump float;                                                                 
#endif       
#define HBAO_RANDOM_ELEMENTS 16                                                          
struct HBAOData                                                                          
{                                                                                        
  vec4    projInfo;
  vec2    invFullRes;                                                          
  vec2    jitters[HBAO_RANDOM_ELEMENTS];
  float   radToScreen;                                                         
  float   rad2;                                                                
  float   negInvRad2;                                                          
  float   NDotVBias;                                                           
  float   AOMult;                                                              
  float   powExp;                                                              
};                                                                                       
                                                                                         
layout(std140) uniform controlBuffer                                                     
{                                                                                        
    HBAOData control;                                                                    
};                                                                                       
#define M_PI 3.14159265f                                                                 
//----------------------------------------------------------------------------------     
// tweakables                                                                            
const float  NUM_STEPS = 4;                                                              
const float  NUM_DIRECTIONS = 8; // texRandom/g_Jitter initialization depends on this    
//----------------------------------------------------------------------------------     
layout(binding = 0) uniform sampler2D sampler0;                                          
//----------------------------------------------------------------------------------     
layout(location = 0, index = 0) out vec4 out_Color;                                      
in vec2 tc;                                                                              
//----------------------------------------------------------------------------------     
vec3 UVToView(vec2 uv, float eye_z)                                                      
{             
  return vec3((uv*control.projInfo.xy+control.projInfo.zw)*eye_z, eye_z);                
}                                                                                        
//----------------------------------------------------------------------------------     
vec3 FetchViewPos(vec2 UV)                                                               
{                                                                                        
  float ViewDepth = textureLod(sampler0, UV, 0).x;                                       
  return UVToView(UV, ViewDepth);                                                        
}                                                                                        
//----------------------------------------------------------------------------------     
vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)                                                   
{                                                                                        
  vec3 V1 = Pr - P;                                                                      
  vec3 V2 = P - Pl;                                                                      
  return (dot(V1, V1) < dot(V2, V2)) ? V1 : V2;                                          
}                                                                                        
//----------------------------------------------------------------------------------     
vec3 ReconstructNormal(vec2 UV, vec3 P)                                                  
{                                                                                        
  vec3 Pr = FetchViewPos(UV + vec2(control.invFullRes.x, 0));                            
  vec3 Pl = FetchViewPos(UV + vec2(-control.invFullRes.x, 0));                           
  vec3 Pt = FetchViewPos(UV + vec2(0, control.invFullRes.y));                            
  vec3 Pb = FetchViewPos(UV + vec2(0, -control.invFullRes.y));                           
  return normalize(cross(MinDiff(P, Pr, Pl), MinDiff(P, Pt, Pb)));                       
}                                                                                        
//----------------------------------------------------------------------------------     
float Falloff(float DistanceSquare)                                                      
{                                                                                        
  // 1 scalar mad instruction                                                            
  return DistanceSquare * control.negInvRad2 + 1.0;                                      
}                                                                                        
//----------------------------------------------------------------------------------     
// P = view-space position at the kernel center                                          
// N = view-space normal at the kernel center                                            
// S = view-space position of the current sample                                         
//----------------------------------------------------------------------------------     
float ComputeAO(vec3 P, vec3 N, vec3 S)                                                  
{                                                                                        
  vec3 V = S - P;                                                                        
  float VdotV = dot(V, V);                                                               
  float NdotV = dot(N, V) * 1.0 / sqrt(VdotV);                                           
  // Use saturate(x) instead of max(x,0.f) because that is faster on Kepler              
  return clamp(NdotV - control.NDotVBias, 0, 1) * clamp(Falloff(VdotV), 0, 1);           
}                                                                                        
//----------------------------------------------------------------------------------     
float ComputeCoarseAO(vec2 FullResUV, float RadPix, vec2 Rand, vec3 ViewPos, vec3 ViewNor)
{                                                                                         
  // Divide by NUM_STEPS+1 so that the farthest samples are not fully attenuated          
  float StepSizePixels = RadPix / (NUM_STEPS + 1);                                        
  const float Alpha = 2.0 * M_PI / NUM_DIRECTIONS;                                        
  float AO = 0;                                                                           
  for (float DirectionIndex = 0; DirectionIndex < NUM_DIRECTIONS; ++DirectionIndex)       
  {                                                                                       
    float Angle = Alpha * DirectionIndex+ Rand.x;                                         
    // Compute normalized 2D direction                                                    
    vec2 Direction = vec2(cos(Angle), sin(Angle));                                        
    // Jitter starting sample within the first step                                       
    float RayPixels = (Rand.y * StepSizePixels + 1.0);                                    
    for (float StepIndex = 0; StepIndex < NUM_STEPS; ++StepIndex)                         
    {                                                    
      vec2 SnappedUV = round(RayPixels * Direction) * control.invFullRes  + FullResUV;    
      vec3 S = FetchViewPos(SnappedUV);                                                   
      RayPixels += StepSizePixels;                                                        
      AO += ComputeAO(ViewPos, ViewNor, S);                                               
    }                                                                                     
  }                                                                                       
  AO *= control.AOMult / (NUM_DIRECTIONS * NUM_STEPS);                                    
  return clamp(1.0 - AO * 2.0, 0, 1);                                                     
}                                                                                         
                                                                                          
//----------------------------------------------------------------------------------      
void main()                                                                               
{                                                                                         
  vec2 uv = tc;                                                                           
  vec3 ViewPos = FetchViewPos(uv);                                                        
  vec3 ViewNor = -ReconstructNormal(uv, ViewPos);                                         
  // Compute projection of disk of radius control.R into screen space                     
  float RadPix = (control.radToScreen / ViewPos.z);                                       
  // Get jitter vector for the current full-res pixel                                     
                                                                                          
  ivec2 RandInd = ivec2(gl_FragCoord.xy) % ivec2(4.0);                                    

  vec2 Rand = control.jitters[RandInd.y * 4 + RandInd.x];
                                                                                          
  float AO = ComputeCoarseAO(uv, RadPix, Rand, ViewPos, ViewNor);                         
  out_Color = vec4(pow(AO, control.powExp));
})";

////////////////////////////////////////////////////////////////////////////////

const char * screen_apply_SSAO_frag =
R"(#version 430 compatibility                             
#ifdef _ANDROID_                                        
precision mediump float;                               
#endif                                                  
in vec2 tc;                                            
uniform sampler2D sampler0;                            
layout(location=0,index=0) out vec4 out_Color;         
                                                       
void main (void)                                       
{                                                      
   out_Color = vec4(texture(sampler0, tc).x);          
})";

////////////////////////////////////////////////////////////////////////////////

#endif

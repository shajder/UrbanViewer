#ifndef RENDERING_RESOURCES_H
#define RENDERING_RESOURCES_H 1

////////////////////////////////////////////////////////////////////////////////

#include <osg/ref_ptr>
#include <osg/Texture2D>
#include <osg/Light>
#include <osg/Vec3>
#include <osg/Group>
#include <osg/BufferTemplate>
#include <osg/BufferIndexBinding>

////////////////////////////////////////////////////////////////////////////////

constexpr short HBAO_RANDOM_SIZE = 4;
constexpr short HBAO_RANDOM_ELEMENTS = (HBAO_RANDOM_SIZE * HBAO_RANDOM_SIZE);
constexpr short MAX_SAMPLES = 1;

constexpr unsigned SHADOW_MASK = 0x00000001;
constexpr unsigned FADE_MASK = 0x00000200;

#define USE_HBAO 1
#define USE_SHADOW 1
#define HARDCODED_WINDOW_SIZE 1

////////////////////////////////////////////////////////////////////////////////

struct RenderingResources
{
  struct HBAOSetup
  {
    int           samples = 1;
    float         intensity = 1.5f;
    float         bias = 0.1f;
    float         radius = 2.0f;
    float         blurSharpness = 40.0f;
    bool          blur = true;
    bool          ortho = false;
  };

  HBAOSetup             hbaoSetup;

  const int numLayers = 4;

  osg::ref_ptr<osg::Texture2D> culture1;
  osg::ref_ptr<osg::Texture2D> culture2;
  osg::ref_ptr<osg::Texture2D> culture3;

  osg::ref_ptr<osg::Light> sunLight;
  osg::ref_ptr<osg::Group> sceneRoot;

  osg::Vec3d sunPos;

  struct HBAOData
  {
    float             radToScreen;        // radius
    float             rad2;               // 1/radius
    float             negInvRad2;         // radius * radius
    float             NDotVBias;
    osg::Vec2         invFullRes;
    osg::Vec2         invQuarterRes;
    float             AOMult;
    float             powExp;
    osg::Vec4         projInfo;
    osg::Vec2         jitters[HBAO_RANDOM_SIZE * HBAO_RANDOM_SIZE];
  };

  osg::ref_ptr< osg::BufferTemplate< HBAOData > >   hbaoBuf;
  osg::ref_ptr<osg::UniformBufferObject>            hbaoUBO;
  osg::ref_ptr<osg::UniformBufferBinding>           hbaoUBB;

  osg::ref_ptr<osg::Uniform>                        clipInfoUnif;
  osg::ref_ptr<osg::Uniform>                        projInfo;
  osg::ref_ptr<osg::Uniform>                        InvFullResolution;

  osg::Vec2f  hbaoRandom[HBAO_RANDOM_ELEMENTS * MAX_SAMPLES];

  int ssao_down_w = 0;
  int ssao_down_h = 0;

  osg::ref_ptr<osgShadow::PSSMLayered>              pssm;

};



#endif

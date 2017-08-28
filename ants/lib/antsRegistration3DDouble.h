#include "antsRegistrationTemplateHeader.h"
#include "antsCommandLineParser.h"
#include "antsCommandLineOption.h"
namespace ants {

//Instantiate the 3DDouble version
int antsRegistration3DDouble(ParserType::Pointer & parser)
{
    return  DoRegistration<double, 3>( parser );
}

} //end namespace ants

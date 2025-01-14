#pragma once

#ifdef DEBUG
    #define DEBUG_LOG(x) std::cout << x << std::endl
#else
    #define DEBUG_LOG(x) // Do nothing
#endif

! RUN: %python %S/../../Semantics/test_errors.py %s %flang_fc1 -fompss-2

SUBROUTINE P(I)                                                                                                                                         
        CHARACTER :: S(10)                                                                                                                              
        !$OSS TASK                                                                                                                                      
        !ERROR: CHARACTER data type is not supported
        s = "TEST"                                                                                                                                      
        !$OSS END TASK                                                                                                                                  
END   
